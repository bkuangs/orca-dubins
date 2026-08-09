"""
Smoke tests for the simulation harness and remaining algorithm stubs.

Avoidance behavior is covered by focused ORCA and primitive-planner tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from orca_dubins.dynamics import integrate, max_turn_rate
from orca_dubins.planners import (
    PreferredVelocityPlanner,
    ReachableOrcaPlanner,
)
from orca_dubins.simulation import SCENARIOS, World
from orca_dubins.types import AircraftParams, AircraftState, vec2


def test_scenarios_build():
    for factory in SCENARIOS.values():
        agents = factory()
        assert len(agents) >= 2


def test_dynamics_constant_speed():
    params = AircraftParams(speed=20.0, max_bank_angle=np.deg2rad(30), radius=5.0)
    state = AircraftState(position=vec2(0.0, 0.0), heading=0.0)
    rate = max_turn_rate(params)
    nxt = integrate(state, rate, params.speed, 0.1)
    # Displacement magnitude over dt should equal speed*dt (constant speed).
    step_len = float(np.linalg.norm(nxt.position - state.position))
    assert step_len == pytest.approx(params.speed * 0.1, rel=1e-3)


def test_baseline_world_runs():
    world = World(agents=SCENARIOS["head_on"](), planner=PreferredVelocityPlanner(), dt=0.1)
    history = world.run(50)
    assert len(history) == 51  # initial frame + 50 steps
    for snap in history:
        for pos in snap.positions.values():
            assert pos.shape == (2,)


def test_reachable_orca_planner_not_implemented():
    agents = SCENARIOS["crossing"]()
    ego, *neighbors = agents
    with pytest.raises(NotImplementedError):
        ReachableOrcaPlanner().compute_velocity(
            ego,
            neighbors,
            ego.preferred_velocity(),
            3.0,
            0.1,
        )
