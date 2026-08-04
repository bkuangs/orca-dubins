"""Ready-made test scenarios for the fixed-wing avoidance prototype.

Each factory returns a list of :class:`Agent`. Combine with a planner to build a
:class:`orca_dubins.simulation.World`. These are the classic stress tests for
reciprocal avoidance: head-on, right-angle crossing, and an antipodal swarm
(everyone flying across a circle to the opposite side).
"""

from __future__ import annotations

import numpy as np

from ..types import Agent, AircraftParams, AircraftState, vec2

DEFAULT_PARAMS = AircraftParams(
    speed=20.0,           # m/s
    max_bank_angle=np.deg2rad(30.0),
    radius=5.0,           # m
)


def _agent(agent_id: str, pos, heading: float, goal, params: AircraftParams) -> Agent:
    return Agent(
        id=agent_id,
        params=params,
        state=AircraftState(position=np.asarray(pos, dtype=float), heading=heading),
        goal=np.asarray(goal, dtype=float),
    )


def head_on(separation: float = 400.0, params: AircraftParams | None = None) -> list[Agent]:
    """Two aircraft flying directly at each other along the x-axis."""
    p = params or DEFAULT_PARAMS
    return [
        _agent("A", vec2(-separation / 2, 0.0), 0.0, vec2(separation / 2, 0.0), p),
        _agent("B", vec2(separation / 2, 0.0), np.pi, vec2(-separation / 2, 0.0), p),
    ]


def crossing(separation: float = 400.0, params: AircraftParams | None = None) -> list[Agent]:
    """Two aircraft on perpendicular courses meeting at the origin."""
    p = params or DEFAULT_PARAMS
    h = separation / 2
    return [
        _agent("A", vec2(-h, 0.0), 0.0, vec2(h, 0.0), p),
        _agent("B", vec2(0.0, -h), np.pi / 2, vec2(0.0, h), p),
    ]


def swarm_circle(
    n: int = 8, radius: float = 300.0, params: AircraftParams | None = None
) -> list[Agent]:
    """``n`` aircraft evenly on a circle, each flying to the antipodal point."""
    p = params or DEFAULT_PARAMS
    agents: list[Agent] = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        pos = vec2(radius * np.cos(theta), radius * np.sin(theta))
        goal = -pos
        heading = float(np.arctan2(goal[1] - pos[1], goal[0] - pos[0]))
        agents.append(_agent(f"A{i}", pos, heading, goal, p))
    return agents


SCENARIOS = {
    "head_on": head_on,
    "crossing": crossing,
    "swarm_circle": swarm_circle,
}
