"""Ready-made test scenarios for the fixed-wing avoidance prototype.

Each factory returns a list of :class:`Agent`. Combine with a planner to build a
:class:`orca_dubins.simulation.World`. These are the classic stress tests for
reciprocal avoidance: head-on, right-angle crossing, and an antipodal swarm
(everyone flying across a circle to the opposite side), plus a seeded random
swarm.
"""

from __future__ import annotations

import numpy as np

from ..types import Agent, AircraftParams, AircraftState, vec2
from .guidance import FormationGuidance, LeaderPathGuidance, SlotSwap

DEFAULT_PARAMS = AircraftParams(
    speed=20.0,           # m/s
    max_bank_angle=np.deg2rad(30.0),
    radius=5.0,           # m
)

FORMATION_PARAMS = AircraftParams(
    speed=20.0,
    max_bank_angle=np.deg2rad(30.0),
    radius=5.0,
    safety_margin=1.5,
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


def swarm_random(
    n: int = 8,
    extent: float = 300.0,
    min_separation: float = 50.0,
    seed: int | None = 0,
    params: AircraftParams | None = None,
) -> list[Agent]:
    """Create a seeded swarm with separated random crossing routes and headings."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if extent <= 0.0:
        raise ValueError("extent must be positive")
    if min_separation <= 0.0:
        raise ValueError("min_separation must be positive")

    p = params or DEFAULT_PARAMS
    if min_separation < 2.0 * p.avoidance_radius:
        raise ValueError("min_separation must be at least the collision diameter")

    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []
    endpoints: list[np.ndarray] = []
    max_attempts = 10_000
    for _ in range(max_attempts):
        angle = rng.uniform(-np.pi, np.pi)
        distance = rng.uniform(0.85 * extent, extent)
        start = distance * np.array([np.cos(angle), np.sin(angle)])
        goal = -start
        route_endpoints = (start, goal)
        if (
            np.linalg.norm(start - goal) >= min_separation
            and all(
                np.linalg.norm(candidate - existing) >= min_separation
                for candidate in route_endpoints
                for existing in endpoints
            )
        ):
            starts.append(start)
            endpoints.extend(route_endpoints)
            if len(starts) == n:
                break
    else:
        raise ValueError(
            "could not place separated crossing routes; increase extent or reduce density"
        )

    heading_offsets = rng.uniform(-np.pi / 12.0, np.pi / 12.0, size=n)
    return [
        _agent(
            f"A{i}",
            starts[i],
            float(np.arctan2(-starts[i][1], -starts[i][0]) + heading_offsets[i]),
            -starts[i],
            p,
        )
        for i in range(n)
    ]


def formation_slots() -> dict[str, np.ndarray]:
    """Return 19 follower offsets for the demonstration V formation."""
    slots: dict[str, np.ndarray] = {}
    follower_index = 1
    for row in range(1, 10):
        slots[f"F{follower_index}"] = vec2(-30.0 * row, 13.0 * row)
        slots[f"F{follower_index + 1}"] = vec2(-30.0 * row, -13.0 * row)
        follower_index += 2
    slots["F19"] = vec2(-300.0, 0.0)
    return slots


def formation_goal_state() -> AircraftState:
    """Return the leader's terminal pose for the formation demonstration."""
    return AircraftState(position=vec2(300.0, 150.0), heading=np.pi / 4.0)


def formation_slot_swaps() -> tuple[SlotSwap, ...]:
    """Return three waves of mirrored slot exchanges during formation flight."""
    waves = (
        (8.0, (1, 4, 7)),
        (14.0, (2, 5, 8)),
        (20.0, (3, 6, 9)),
    )
    return tuple(
        SlotSwap(
            time=time,
            first_id=f"F{2 * row - 1}",
            second_id=f"F{2 * row}",
        )
        for time, rows in waves
        for row in rows
    )


def swarm_formation(
    seed: int | None = 0,
    params: AircraftParams | None = None,
) -> list[Agent]:
    """Create one leader and 19 followers that cross while assembling a V."""
    p = params or FORMATION_PARAMS
    rng = np.random.default_rng(seed)
    leader_position = vec2(-300.0, -150.0)
    leader_heading = 0.0
    goal_state = formation_goal_state()
    slots = formation_slots()
    agents = [
        _agent(
            "L",
            leader_position,
            leader_heading,
            goal_state.position,
            p,
        )
    ]
    occupied = [leader_position]

    goal_rotation = np.array([
        [np.cos(goal_state.heading), -np.sin(goal_state.heading)],
        [np.sin(goal_state.heading), np.cos(goal_state.heading)],
    ])
    for follower_id, offset in slots.items():
        nominal_start = leader_position + offset
        for _ in range(1_000):
            start = nominal_start + rng.uniform(-3.0, 3.0, size=2)
            if all(
                np.linalg.norm(start - position) >= 2.4 * p.avoidance_radius
                for position in occupied
            ):
                break
        else:
            raise ValueError("could not place separated formation followers")

        occupied.append(start)
        heading = leader_heading + rng.uniform(-np.pi / 12.0, np.pi / 12.0)
        goal = goal_state.position + goal_rotation @ offset
        agents.append(_agent(follower_id, start, float(heading), goal, p))

    return agents


def swarm_formation_guidance() -> FormationGuidance:
    """Build guidance for :func:`swarm_formation`."""
    return FormationGuidance(
        leader_guidance=LeaderPathGuidance(
            leader_id="L",
            goal_state=formation_goal_state(),
        ),
        slots=formation_slots(),
        position_gain=2.0,
        slot_swaps=formation_slot_swaps(),
    )


SCENARIOS = {
    "head_on": head_on,
    "crossing": crossing,
    "swarm_circle": swarm_circle,
    "swarm_random": swarm_random,
    "swarm_formation": swarm_formation,
}
