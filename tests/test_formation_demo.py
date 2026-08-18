"""End-to-end validation for the leader/follower formation demo."""

from __future__ import annotations

import pytest

from orca_dubins.planners import PreferredVelocityPlanner, ReachableOrcaPlanner
from orca_dubins.simulation import (
    World,
    assigned_slot_rmse,
    avoidance_intervention_count,
    leader_cross_track_rmse,
    minimum_pairwise_separation,
    swarm_formation,
    swarm_formation_guidance,
)


@pytest.fixture(scope="module")
def formation_run():
    agents = swarm_formation(seed=0)
    guidance = swarm_formation_guidance()
    world = World(
        agents=agents,
        planner=ReachableOrcaPlanner(),
        guidance=guidance,
        record_diagnostics=True,
        dt=0.1,
        horizon=3.0,
        goal_tolerance=5.0,
    )
    world.run_until_complete(600)
    return world, guidance


def test_formation_demo_has_twenty_agents():
    assert len(swarm_formation(seed=0)) == 20


def test_formation_demo_creates_collision_risk_without_avoidance():
    world = World(
        agents=swarm_formation(seed=0),
        planner=PreferredVelocityPlanner(),
        guidance=swarm_formation_guidance(),
        dt=0.1,
        horizon=3.0,
        goal_tolerance=5.0,
    )

    world.run_until_complete(600)

    assert minimum_pairwise_separation(world.history) < 10.0


def test_formation_demo_completes_without_overlap(formation_run):
    world, _ = formation_run

    assert world.all_arrived()
    combined_radius = 2.0 * world.agents[0].params.avoidance_radius
    assert minimum_pairwise_separation(world.history) >= combined_radius
    assert avoidance_intervention_count(world.history) > 100


def test_formation_demo_tracks_leader_route(formation_run):
    world, guidance = formation_run

    error = leader_cross_track_rmse(
        world.history,
        guidance.leader_guidance,
    )
    assert error < 3.0


def test_formation_demo_tracks_follower_slots(formation_run):
    world, guidance = formation_run

    error = assigned_slot_rmse(
        world.history,
        guidance,
        start_index=len(world.history) // 4,
    )
    assert error < 110.0
