#!/usr/bin/env python
"""
Uses the DES engine to compute the Fibonacci sequence
"""
from desh.engine import State, Event, Engine, Priority
from desh.render import c_out, Palette
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class FiboState(State):
    x_i : int
    x_ii : int

@dataclass(frozen=True)
class Update(Event):
    def execute(self, state: FiboState) -> tuple[FiboState, list[Event]]:
        new_x_i = state.x_ii
        new_x_ii = state.x_i + state.x_ii
        return replace(state, x_i=new_x_i, x_ii=new_x_ii), [Echo(), Update()]

@dataclass(frozen=True)
class Echo(Event):
    priority = Priority.HIGH
    def execute(self, state: FiboState) -> tuple[FiboState, list[Event]]:
        print(c_out(Palette.CHROME, f"x_i: {state.x_i} x_ii: {state.x_ii} ratio: {state.x_ii / state.x_i if state.x_i != 0 else 0}"))
        return state, []


if __name__ == "__main__":
    engine = Engine[FiboState](debug=True)
    state = FiboState(x_i=0, x_ii=1)
    max_steps = 20
    for i, s in enumerate(engine.iter(state, [Update()])):
        if i == max_steps:
            break
