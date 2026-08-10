"""Behavioral tests for ORCA over the Dubins-reachable velocity arc."""

from __future__ import annotations

import numpy as np
import pytest

from orca_dubins.dynamics import max_turn_rate, wrap_angle
from orca_dubins.planners import ReachableOrcaPlanner
from orca_dubins.simulation import SCENARIOS, World
from orca_dubins.types import Agent, AircraftParams, AircraftState, vec2


def _agent(heading: float = 0.0) -> Agent:
    return Agent(
        id="A",
        params=AircraftParams(
            speed=20.0,
            max_bank_angle=float(np.deg2rad(30.0)),
            radius=5.0,
        ),
        state=AircraftState(position=vec2(0.0, 0.0), heading=heading),
        goal=vec2(100.0, 0.0),
    )


def _heading(velocity: np.ndarray) -> float:
    return float(np.arctan2(velocity[1], velocity[0]))


def test_reachable_planner_returns_next_step_of_horizon_turn():
    ego = _agent()
    planner = ReachableOrcaPlanner()
    horizon = 3.0
    dt = 0.1
    rate_limit = float(max_turn_rate(ego.params))
    terminal_offset = 0.5 * rate_limit * horizon
    preferred_velocity = ego.params.speed * np.array(
        [np.cos(terminal_offset), np.sin(terminal_offset)]
    )

    command = planner.compute_velocity(
        ego,
        [],
        preferred_velocity,
        horizon,
        dt,
    )

    commanded_offset = wrap_angle(_heading(command) - ego.state.heading)
    assert np.linalg.norm(command) == pytest.approx(ego.params.speed)
    assert commanded_offset == pytest.approx(0.5 * rate_limit * dt)


def test_reachable_planner_clips_preference_to_reachable_arc():
    ego = _agent()
    planner = ReachableOrcaPlanner()
    horizon = 3.0
    dt = 0.1
    rate_limit = float(max_turn_rate(ego.params))
    unreachable_offset = 2.0 * rate_limit * horizon
    preferred_velocity = ego.params.speed * np.array(
        [np.cos(unreachable_offset), np.sin(unreachable_offset)]
    )

    command = planner.compute_velocity(
        ego,
        [],
        preferred_velocity,
        horizon,
        dt,
    )

    commanded_offset = wrap_angle(_heading(command) - ego.state.heading)
    assert commanded_offset == pytest.approx(rate_limit * dt)


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_reachable_planner_rejects_nonpositive_timestep(dt: float):
    ego = _agent()

    with pytest.raises(ValueError, match="dt"):
        ReachableOrcaPlanner().compute_velocity(
            ego,
            [],
            ego.preferred_velocity(),
            horizon=3.0,
            dt=dt,
        )


@pytest.mark.parametrize("horizon", [0.0, -1.0])
def test_reachable_planner_rejects_nonpositive_horizon(horizon: float):
    ego = _agent()

    with pytest.raises(ValueError, match="horizon"):
        ReachableOrcaPlanner().compute_velocity(
            ego,
            [],
            ego.preferred_velocity(),
            horizon=horizon,
            dt=0.1,
        )


def test_reachable_planner_rejects_horizon_shorter_than_timestep():
    ego = _agent()

    with pytest.raises(ValueError, match="horizon"):
        ReachableOrcaPlanner().compute_velocity(
            ego,
            [],
            ego.preferred_velocity(),
            horizon=0.05,
            dt=0.1,
        )


def test_reachable_planner_returns_finite_fallback_when_safe_arc_is_empty():
    ego = _agent()
    neighbor = Agent(
        id="B",
        params=ego.params,
        state=AircraftState(position=vec2(0.1, 0.0), heading=np.pi),
        goal=vec2(-100.0, 0.0),
    )

    command = ReachableOrcaPlanner().compute_velocity(
        ego,
        [neighbor],
        ego.preferred_velocity(),
        horizon=3.0,
        dt=0.1,
    )

    assert np.all(np.isfinite(command))
    assert np.linalg.norm(command) == pytest.approx(ego.params.speed)


def test_reachable_planner_runs_in_world_with_finite_constant_speed_commands():
    agents = SCENARIOS["crossing"]()
    world = World(
        agents=agents,
        planner=ReachableOrcaPlanner(),
        dt=0.1,
        horizon=3.0,
    )

    history = world.run(25)

    assert len(history) == 26
    assert all(
        np.all(np.isfinite(command))
        for snapshot in history
        for command in snapshot.commanded.values()
    )
    assert all(
        np.linalg.norm(command) == pytest.approx(agent.params.speed)
        for snapshot in history
        for agent in agents
        if (command := snapshot.commanded.get(agent.id)) is not None
    )
