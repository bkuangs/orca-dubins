"""Integration tests for the discrete primitive ORCA planner."""

from __future__ import annotations

import numpy as np
import pytest

from orca_dubins.dynamics import max_turn_rate, steer_toward_velocity
from orca_dubins.planners import PrimitiveOrcaPlanner
from orca_dubins.simulation import SCENARIOS, World


def test_planner_commands_the_selected_primitive_for_the_next_step():
    ego = SCENARIOS["head_on"]()[0]
    planner = PrimitiveOrcaPlanner(n_primitives=5)
    horizon = 3.0
    dt = 0.1
    rate_limit = float(max_turn_rate(ego.params))
    preferred_heading = ego.state.heading + rate_limit * horizon
    preferred_velocity = ego.params.speed * np.array(
        [np.cos(preferred_heading), np.sin(preferred_heading)]
    )

    commanded_velocity = planner.compute_velocity(
        ego,
        [],
        preferred_velocity,
        horizon,
        dt,
    )
    commanded_turn_rate = steer_toward_velocity(
        ego.state.heading,
        commanded_velocity,
        rate_limit,
        dt,
    )

    assert commanded_turn_rate == pytest.approx(rate_limit)


@pytest.mark.parametrize("scenario", ["head_on", "crossing"])
def test_primitive_planner_avoids_overlap_in_two_agent_scenarios(scenario: str):
    agents = SCENARIOS[scenario]()
    world = World(
        agents=agents,
        planner=PrimitiveOrcaPlanner(),
        dt=0.1,
        horizon=3.0,
    )

    history = world.run(200)
    minimum_separation = min(
        float(
            np.linalg.norm(
                snapshot.positions[agents[0].id]
                - snapshot.positions[agents[1].id]
            )
        )
        for snapshot in history
    )
    combined_radius = agents[0].params.radius + agents[1].params.radius

    assert minimum_separation >= combined_radius


def test_planner_rejects_horizon_shorter_than_control_timestep():
    ego = SCENARIOS["head_on"]()[0]

    with pytest.raises(ValueError, match="horizon"):
        PrimitiveOrcaPlanner().compute_velocity(
            ego,
            [],
            ego.preferred_velocity(),
            horizon=0.05,
            dt=0.1,
        )


def test_swarm_circle_continues_when_agents_begin_to_overlap():
    agents = SCENARIOS["swarm_circle"]()
    world = World(
        agents=agents,
        planner=PrimitiveOrcaPlanner(),
        dt=0.1,
        horizon=3.0,
    )

    history = world.run(300)

    assert len(history) == 301
    assert all(
        np.all(np.isfinite(position))
        for snapshot in history
        for position in snapshot.positions.values()
    )
