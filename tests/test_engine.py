"""Contract tests for the generic desh.engine.Engine[S] — independent of
desh_chat. Uses small synthetic State/Event pairs (a counter, and a bounded
fibonacci-shaped pair matching examples/desh-fibonacci.py's spirit but
terminating on its own, since the example's own Update never stops and
would hang run()/trace()).

Covers: pure-state transactionality, the drain/propagate/halt exit
taxonomy, on_error/on_interrupt handlers returning events only, the
des-log RUN_START/RUN_END bracket (including a fatal run), priority
ordering, and run/iter/trace equivalence.
"""
import io
import json
from dataclasses import dataclass, replace

import pytest

from desh.engine import Engine, Event, HandlerError, Priority, State


# ---------------------
# Synthetic state/events
# ---------------------

@dataclass(frozen=True)
class CounterState(State):
    value: int
    log: tuple[str, ...] = ()


@dataclass(frozen=True)
class Increment(Event):
    def execute(self, state: CounterState) -> tuple[CounterState, list[Event]]:
        return replace(state, value=state.value + 1), []


@dataclass(frozen=True)
class Poison(Event):
    """Raises before ever returning a new state — nothing it computes should
    become visible to the caller.
    """
    def execute(self, state: CounterState) -> tuple[CounterState, list[Event]]:
        replace(state, value=999)  # computed, deliberately never returned
        raise ValueError("boom")


@dataclass(frozen=True)
class RaiseInterrupt(Event):
    def execute(self, state: CounterState) -> tuple[CounterState, list[Event]]:
        raise KeyboardInterrupt()


@dataclass(frozen=True)
class Mark(Event):
    label: str
    def execute(self, state: CounterState) -> tuple[CounterState, list[Event]]:
        return replace(state, log=state.log + (self.label,)), []


class HighMark(Mark):
    priority = Priority.HIGH


@dataclass(frozen=True)
class FiboState(State):
    x_i: int
    x_ii: int
    steps_left: int


@dataclass(frozen=True)
class FiboStep(Event):
    def execute(self, state: FiboState) -> tuple[FiboState, list[Event]]:
        new_state = replace(state, x_i=state.x_ii, x_ii=state.x_i + state.x_ii, steps_left=state.steps_left - 1)
        return new_state, [FiboStep()] if new_state.steps_left > 0 else []


def read_log(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().splitlines()]


# ---------------------
# Pure-state transactionality
# ---------------------

class TestTransactionality:
    def test_a_raising_step_leaves_the_last_committed_state_visible(self):
        engine = Engine[CounterState]()
        seen = []
        with pytest.raises(ValueError):
            for s in engine.iter(CounterState(value=0), seed=[Increment(), Increment(), Poison(), Increment()]):
                seen.append(s)
        assert seen[-1].value == 2  # both real increments landed, poison never committed, the trailing Increment never ran

    def test_a_state_computed_inside_a_raising_step_never_becomes_visible(self):
        # Poison computes replace(state, value=999) and then raises without
        # returning it — the engine's own state variable must never have
        # been reassigned to it.
        engine = Engine[CounterState]()
        with pytest.raises(ValueError):
            engine.run(CounterState(value=0), seed=[Poison()])
        # nothing to read post-raise from run() itself (it doesn't return on
        # exception) — the guarantee is exercised via iter() above instead.

    def test_on_error_recovery_does_not_resurrect_the_poisoned_computation(self):
        def on_error(ev, ex, s):
            return [Increment()]
        final = Engine[CounterState](on_error=on_error).run(CounterState(value=0), seed=[Increment(), Poison()])
        # value=1 from the real Increment, then Poison's failed step
        # contributes nothing, then on_error's own Increment lands: 2.
        # If the poisoned replace(state, value=999) had leaked through, this
        # would read 1000, not 2.
        assert final.value == 2


# ---------------------
# Exit taxonomy
# ---------------------

class TestExitTaxonomy:
    def test_drain_on_normal_completion(self):
        buf = io.StringIO()
        Engine[CounterState](des_log=buf).run(CounterState(value=0), seed=[Increment(), Increment()])
        assert read_log(buf)[-1]["exit_mode"] == "drain"

    def test_propagate_on_unhandled_exception(self):
        buf = io.StringIO()
        with pytest.raises(ValueError):
            Engine[CounterState](des_log=buf).run(CounterState(value=0), seed=[Poison()])
        assert read_log(buf)[-1]["exit_mode"] == "propagate"

    def test_propagate_on_keyboard_interrupt_with_no_interrupt_handler(self):
        buf = io.StringIO()
        with pytest.raises(KeyboardInterrupt):
            Engine[CounterState](des_log=buf).run(CounterState(value=0), seed=[RaiseInterrupt()])
        assert read_log(buf)[-1]["exit_mode"] == "propagate"

    def test_halt_when_the_error_handler_itself_raises(self):
        buf = io.StringIO()
        def broken_handler(ev, ex, s):
            raise RuntimeError("handler is broken")
        with pytest.raises(HandlerError):
            Engine[CounterState](des_log=buf, on_error=broken_handler).run(CounterState(value=0), seed=[Poison()])
        assert read_log(buf)[-1]["exit_mode"] == "halt"

    def test_halt_when_the_interrupt_handler_itself_raises(self):
        buf = io.StringIO()
        def broken_handler(s):
            raise RuntimeError("handler is broken")
        with pytest.raises(HandlerError):
            Engine[CounterState](des_log=buf, on_interrupt=broken_handler).run(CounterState(value=0), seed=[RaiseInterrupt()])
        assert read_log(buf)[-1]["exit_mode"] == "halt"


# ---------------------
# Handler contract: on_error / on_interrupt return events only
# ---------------------

class TestHandlers:
    def test_on_error_pushes_the_events_it_returns(self):
        seen_errors = []
        def on_error(ev, ex, s):
            seen_errors.append((type(ev).__name__, type(ex).__name__))
            return [Increment(), Increment()]
        final = Engine[CounterState](on_error=on_error).run(CounterState(value=0), seed=[Poison()])
        assert final.value == 2
        assert seen_errors == [("Poison", "ValueError")]

    def test_on_interrupt_pushes_the_events_it_returns(self):
        final = Engine[CounterState](on_interrupt=lambda s: [Increment()]).run(
            CounterState(value=0), seed=[RaiseInterrupt()])
        assert final.value == 1

    def test_no_error_handler_propagates_unhandled(self):
        with pytest.raises(ValueError):
            Engine[CounterState]().run(CounterState(value=0), seed=[Poison()])

    def test_no_interrupt_handler_propagates_keyboard_interrupt(self):
        with pytest.raises(KeyboardInterrupt):
            Engine[CounterState]().run(CounterState(value=0), seed=[RaiseInterrupt()])


# ---------------------
# des-log RUN_START / RUN_END bracket
# ---------------------

class TestDesLogBracket:
    def test_brackets_a_normal_run(self):
        buf = io.StringIO()
        Engine[CounterState](des_log=buf).run(CounterState(value=0), seed=[Increment()])
        records = read_log(buf)
        assert records[0]["event"] == "RUN_START"
        assert records[-1]["event"] == "RUN_END"

    def test_brackets_a_fatal_run_too(self):
        buf = io.StringIO()
        with pytest.raises(ValueError):
            Engine[CounterState](des_log=buf).run(CounterState(value=0), seed=[Poison()])
        records = read_log(buf)
        assert records[0]["event"] == "RUN_START"
        assert records[-1]["event"] == "RUN_END"
        assert records[-1]["exit_mode"] == "propagate"

    def test_log_header_fields_are_merged_into_run_start(self):
        buf = io.StringIO()
        Engine[CounterState](des_log=buf).run(CounterState(value=0), seed=[Increment()], log_header={"model": "test-model"})
        assert read_log(buf)[0]["model"] == "test-model"

    def test_step_records_carry_event_name_and_emitted_names(self):
        buf = io.StringIO()
        Engine[CounterState](des_log=buf).run(CounterState(value=0), seed=[Mark("a")])
        step_records = [r for r in read_log(buf) if r["event"] not in ("RUN_START", "RUN_END")]
        assert len(step_records) == 1
        assert step_records[0]["event"] == "Mark"
        assert step_records[0]["emitted"] == []
        assert step_records[0]["outcome"] == "ok"


# ---------------------
# Priority ordering
# ---------------------

class TestPriority:
    def test_high_priority_event_runs_before_a_normal_one_queued_earlier(self):
        final = Engine[CounterState]().run(CounterState(value=0), seed=[Mark("normal"), HighMark("high")])
        assert final.log == ("high", "normal")

    def test_same_priority_events_run_in_seed_order(self):
        final = Engine[CounterState]().run(CounterState(value=0), seed=[Mark("first"), Mark("second")])
        assert final.log == ("first", "second")


# ---------------------
# run / iter / trace equivalence
# ---------------------

class TestRunIterTraceEquivalence:
    def test_all_three_agree_on_the_final_state(self):
        seed_state = FiboState(x_i=0, x_ii=1, steps_left=10)
        run_final = Engine[FiboState]().run(seed_state, seed=[FiboStep()])
        iter_final = list(Engine[FiboState]().iter(seed_state, seed=[FiboStep()]))[-1]
        trace_final = Engine[FiboState]().trace(seed_state, seed=[FiboStep()])[-1]
        assert run_final == iter_final == trace_final
        assert run_final.x_i == 55 and run_final.x_ii == 89  # 10 steps of fibonacci from (0, 1)

    def test_iter_and_trace_agree_on_every_intermediate_state(self):
        seed_state = FiboState(x_i=0, x_ii=1, steps_left=10)
        iter_states = list(Engine[FiboState]().iter(seed_state, seed=[FiboStep()]))
        trace_states = list(Engine[FiboState]().trace(seed_state, seed=[FiboStep()]))
        assert iter_states == trace_states
        assert len(trace_states) == 10  # one state per FiboStep executed
        assert trace_states[-1].steps_left == 0

    def test_trace_returns_a_tuple(self):
        result = Engine[FiboState]().trace(FiboState(x_i=0, x_ii=1, steps_left=3), seed=[FiboStep()])
        assert isinstance(result, tuple)
