"""
Abstract Discrete-Event-Simulation engine.
Defines:
- State: abstract base class for simulation state (Empty ABC)
- Event and priorities: thin abstract base class for simulation events that carries priority and abstract execute method.
- Engine: DES engine to run the simulation. 
    - Usage: After init, call `run()`, `stream()`, or `trace()` with a run ID, initial state, and seed events.
        - run(): runs the simulation until the queue is empty and returns the final state.
        - stream(): runs the simulation until the queue is empty and yields the state after each step
        - trace(): runs the simulation until the queue is empty and yields the state after each step
    - Transactional guarantee: 
        Each step is executed atomically, and the state is only updated if the step completes successfully. 
        If a step raises an exception (on any path), the state visible to the caller is the last commited one.
        A failed step mutates nothing.
    - Handlers: 
        - on_interrupt: called when the simulation step raises KeyboardInterrupt
        - on_error: called for any other Exception during a simulation step.
        - all handlers must return a list of events to be pushed to the queue.
        - BaseExceptions that are not Exception or KeyboardInterrupt are not handled and will propagate with outcome "unhandled".
    - Logging: if active, logs to a jsonl file with the following fields:
        - run: run_id
        - ts: timestamp of the step
        - n: step number   
        - depth: queue depth after the step
        - event: name of the event class
        - payload: repr of the event (bounded)
        - emitted: list of names of the emitted events
        - dur_ms: duration of the step in milliseconds
        - outcome: one of "ok", "interrupted", "recovered", "step_error", "handler_error", "unhandled"
    - Debug: if active, prints to stderr a human-readable log of the step `↪ Event(...) → [NewEvents, ...] depth=<queue size> <step duration>ms`
    - Raises:
        - SimulationError: raised by engine on contract violations, e.g. empty queue.
        - HandlerError(SimulationError): raised by engine when a given handler fails.
        - KeyboardInterrupt or Exception during step execution will propagate if no handler is provided.
    - Simulation exit condition:
        - drain (queue empty, normal)
        - propagate (unhandled, step_error, no-policy interrupt)
        - halt (HandlerError)

TODO: implement async stream() method to yield state after each step asynchronously.
"""
import heapq
import sys
from  time import time, strftime
import json
from itertools import count
from abc import ABC, abstractmethod
from enum import IntEnum, StrEnum
from typing import Callable, Iterator, TextIO
from desh.render import Palette, c_err, bounded_repr, pretty_log
from uuid import uuid4


# -----------------------
# Abstract base class for states.
# -----------------------

class State(ABC):
    """Abstract base class for state."""
    pass


# -----------------------
# Abstract base class for events.
# -----------------------

class Priority(IntEnum):
    HIGH, NORMAL, LOW = 0, 10, 20

class Event[S: State](ABC):
    """Abstract base class for events."""
    priority: int = Priority.NORMAL

    @abstractmethod
    def execute(self, state: S) -> tuple[S, list[Event]]:
        raise NotImplementedError


# -----------------------
# DES Engine
# -----------------------

class StepOutcome(StrEnum):
    UNHANDLED    = "unhandled"  # bounds outcome and is present in unhandled BaseException\Exception = {SystemExit, GeneratorExit}
    OK           = "ok"
    INTERRUPTED  = "interrupted"
    RECOVERED    = "recovered"
    STEP_ERROR   = "step_error"
    HANDLER_ERROR = "handler_error"

class ExitMode(StrEnum):
    DRAIN       = "drain"       # queue empty, normal exit
    PROPAGATE   = "propagate"   # unhandled exception, propagate
    HALT        = "halt"        # handler error, halt simulation

class SimulationError(Exception):
    pass

class HandlerError(SimulationError):
    pass

class Engine[S: State]:
    """Engine to run the DES simulation."""
    def __init__(self, *, des_log: TextIO | None = None, debug: bool = False,
                    on_error: Callable[[Event, Exception, S], list[Event]] | None = None,
                    on_interrupt: Callable[[S], list[Event]] | None = None):
        self.run_id = ""
        self.queue = []
        self.seq = count()
        self.des_log = des_log
        self.debug = debug
        self.on_error = on_error
        self.on_interrupt = on_interrupt
        self.run_step = 0

    def _push(self, events: list[Event]):
        for event in events:
            heapq.heappush(self.queue, (event.priority, next(self.seq), event))

    def _log_header(self, log_header: dict | None = None):
        if self.des_log:
            header = {**(log_header or {}), "run" : self.run_id, "ts" : time(), "event" : "RUN_START"}
            self.des_log.write(json.dumps(header) + "\n")
            self.des_log.flush()

    def _log_footer(self, exit_mode : ExitMode):
        if self.des_log:
            self.des_log.write(json.dumps({
                    "run" : self.run_id,
                    "ts"  : time(),
                    "event" : "RUN_END",
                    "exit_mode" : exit_mode.value,
                    "steps" : self.run_step,
                    "depth" : len(self.queue)
                    }) + "\n")
            self.des_log.flush()

    def _log_step(self, event: Event, new_events: list[Event], t0: float, t1: float, outcome: StepOutcome):
        log_record = {
            "run": self.run_id,
            "ts": time(),
            "n" : self.run_step,
            "depth" : len(self.queue),
            "event" : type(event).__name__,
            "payload" : bounded_repr.repr(event),
            "emitted" : [type(e).__name__ for e in new_events],
            "dur_ms"  : round((t1 - t0) * 1e3, 1),
            "outcome" : outcome
        }
        if self.debug:
            print(c_err(Palette.DEBUG, pretty_log(log_record)), file=sys.stderr)
        if self.des_log:
            self.des_log.write(json.dumps(log_record) + "\n")
            self.des_log.flush()

    def _step(self, state: S) -> S:
        """Execute the next event in the queue."""
        outcome, new_events = StepOutcome.UNHANDLED, []
        if not self.queue:
            raise SimulationError("No events in queue to process.")
        _, _, event = heapq.heappop(self.queue)
        t0 = time()
        try:
            state, new_events = event.execute(state)
            outcome = StepOutcome.OK
        except KeyboardInterrupt:
            outcome = StepOutcome.INTERRUPTED
            if self.on_interrupt is None:
                raise
            try:
                new_events = self.on_interrupt(state)
            except Exception as handler_error:
                outcome = StepOutcome.HANDLER_ERROR
                raise HandlerError("on_interrupt handler failed") from handler_error
        except Exception as step_error:
            if self.on_error is None:
                outcome = StepOutcome.STEP_ERROR
                raise
            try:
                new_events = self.on_error(event, step_error, state)
                outcome = StepOutcome.RECOVERED
            except Exception as handler_error:
                outcome = StepOutcome.HANDLER_ERROR
                raise HandlerError("on_error handler failed") from handler_error
        finally:
            t1 = time()
            self._log_step(event, new_events, t0, t1, outcome)

        self._push(new_events)
        self.run_step += 1
        return state

    def _simulate(self, state: S, seed: list[Event], *, run_id: str = "", log_header: dict | None = None) -> Iterator[S]:
        """Run the simulation until the queue is empty."""
        if run_id == "" and self.des_log:
            run_id = f"{strftime('%Y%m%d-%H%M%S')}_{uuid4().hex[:6]}"  # Unique run ID
        self.run_id = run_id
        self._log_header(log_header)
        exit_mode = ExitMode.DRAIN
        self.queue.clear()
        self._push(seed)
        try:
            while self.queue:
                state = self._step(state)
                yield state
        except HandlerError:
            exit_mode = ExitMode.HALT
            raise
        except BaseException:
            exit_mode = ExitMode.PROPAGATE
            raise
        finally:
            self._log_footer(exit_mode)
            self.run_id = ""
            self.run_step = 0
            # Not clearing the queue for allowing inspection post-crash

    def run(self, state: S, seed: list[Event], *, run_id: str = "", log_header: dict | None = None) -> S:
        """Run the simulation until the queue is empty and return the final state."""
        for state in self._simulate(state, seed, run_id=run_id, log_header=log_header):
            pass
        return state

    def iter(self, state: S, seed: list[Event], *, run_id: str = "", log_header: dict | None = None) -> Iterator[S]:
        """Run the simulation until the queue is empty and yield the state after each step."""
        yield from self._simulate(state, seed, run_id=run_id, log_header=log_header)

    def trace(self, state: S, seed: list[Event], *, run_id: str = "", log_header: dict | None = None) -> tuple[S, ...]:
        """Run the simulation until the queue is empty and return a tuple of all states after each step."""
        return tuple(self._simulate(state, seed, run_id=run_id, log_header=log_header))
