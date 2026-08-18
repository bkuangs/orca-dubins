"""Behavioral tests for the seeded random crossing swarm."""

from __future__ import annotations

import numpy as np

from orca_dubins.planners import PreferredVelocityPlanner, ReachableOrcaPlanner
from orca_dubins.simulation import (
    World,
    minimum_pairwise_separation,
    swarm_random,
)


def test_random_swarm_is_seeded_and_initially_separated():
    agents = swarm_random(seed=7)
    repeated = swarm_random(seed=7)

    assert len(agents) == 8
    for agent, repeated_agent in zip(agents, repeated, strict=True):
        np.testing.assert_array_equal(
            agent.state.position,
            repeated_agent.state.position,
        )
        np.testing.assert_array_equal(agent.goal, repeated_agent.goal)
        assert agent.state.heading == repeated_agent.state.heading

    for index, agent in enumerate(agents):
        for other in agents[index + 1:]:
            separation = np.linalg.norm(
                agent.state.position - other.state.position
            )
            assert separation >= 50.0


def test_random_swarm_routes_create_collision_risk():
    world = World(
        agents=swarm_random(seed=7),
        planner=PreferredVelocityPlanner(),
        dt=0.1,
    )

    world.run(400)

    assert minimum_pairwise_separation(world.history) < 10.0


def test_reachable_orca_avoids_random_swarm_collisions():
    world = World(
        agents=swarm_random(seed=7),
        planner=ReachableOrcaPlanner(),
        dt=0.1,
    )

    world.run(350)

    assert minimum_pairwise_separation(world.history) >= 10.0
