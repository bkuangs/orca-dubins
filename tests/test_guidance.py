"""Tests for mission guidance integration with the simulation world."""

from __future__ import annotations

import numpy as np

from orca_dubins.dynamics import wrap_angle
from orca_dubins.planners import PreferredVelocityPlanner
from orca_dubins.simulation import (
    FormationGuidance,
    LeaderPathGuidance,
    PointGoalGuidance,
    SlotSwap,
    World,
)
from orca_dubins.types import Agent, AircraftParams, AircraftState, vec2


def _agent(position: np.ndarray, goal: np.ndarray) -> Agent:
    return Agent(
        id="A",
        params=AircraftParams(
            speed=20.0,
            max_bank_angle=float(np.deg2rad(30.0)),
            radius=5.0,
        ),
        state=AircraftState(position=position, heading=0.0),
        goal=goal,
    )


class FixedGuidance:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], float]] = []

    def preferred_velocity(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> np.ndarray:
        self.calls.append((ego.id, tuple(agent.id for agent in agents), time))
        return vec2(0.0, ego.params.speed)

    def is_complete(
        self,
        ego: Agent,
        agents: list[Agent],
        time: float,
        tolerance: float,
    ) -> bool:
        return False


def test_point_goal_guidance_preserves_existing_completion_behavior():
    agent = _agent(vec2(0.0, 0.0), vec2(0.0, 0.0))
    world = World(
        agents=[agent],
        planner=PreferredVelocityPlanner(),
        guidance=PointGoalGuidance(),
    )

    snapshot = world.step()

    assert agent.id not in snapshot.commanded
    np.testing.assert_array_equal(agent.state.position, vec2(0.0, 0.0))


def test_world_uses_injected_mission_guidance():
    agent = _agent(vec2(0.0, 0.0), vec2(0.0, 0.0))
    guidance = FixedGuidance()
    world = World(
        agents=[agent],
        planner=PreferredVelocityPlanner(),
        guidance=guidance,
    )

    snapshot = world.step()

    np.testing.assert_array_equal(
        snapshot.commanded[agent.id],
        vec2(0.0, agent.params.speed),
    )
    assert guidance.calls == [("A", ("A",), 0.0)]


def test_formation_guidance_applies_scheduled_slot_swap():
    goal_state = AircraftState(vec2(50.0, 20.0), np.pi / 4.0)
    guidance = FormationGuidance(
        LeaderPathGuidance("L", goal_state),
        slots={
            "F1": vec2(-10.0, 5.0),
            "F2": vec2(-10.0, -5.0),
        },
        slot_swaps=(SlotSwap(8.0, "F1", "F2"),),
    )

    np.testing.assert_array_equal(
        guidance.slot_offset("F1", time=7.9),
        vec2(-10.0, 5.0),
    )
    np.testing.assert_array_equal(
        guidance.slot_offset("F1", time=8.0),
        vec2(-10.0, -5.0),
    )


def test_leader_path_guidance_reaches_terminal_pose():
    params = AircraftParams(
        speed=5.0,
        max_bank_angle=float(np.deg2rad(30.0)),
        radius=1.0,
    )
    goal_state = AircraftState(vec2(50.0, 30.0), np.pi / 2.0)
    leader = Agent(
        id="L",
        params=params,
        state=AircraftState(vec2(0.0, 0.0), 0.0),
        goal=goal_state.position.copy(),
    )
    guidance = LeaderPathGuidance("L", goal_state)
    world = World(
        agents=[leader],
        planner=PreferredVelocityPlanner(),
        guidance=guidance,
        dt=0.05,
        goal_tolerance=1.0,
    )

    world.run(400)

    assert world.all_arrived()
    assert guidance.path is not None
    assert np.linalg.norm(leader.state.position - goal_state.position) <= 1.0
    assert abs(wrap_angle(leader.state.heading - goal_state.heading)) <= (
        guidance.heading_tolerance
    )


def test_leader_path_guidance_replans_after_large_deviation():
    params = AircraftParams(
        speed=5.0,
        max_bank_angle=float(np.deg2rad(30.0)),
        radius=1.0,
    )
    leader = Agent(
        id="L",
        params=params,
        state=AircraftState(vec2(0.0, 0.0), 0.0),
        goal=vec2(50.0, 30.0),
    )
    guidance = LeaderPathGuidance(
        "L",
        AircraftState(leader.goal.copy(), np.pi / 2.0),
    )
    agents = [leader]
    guidance.preferred_velocity(leader, agents, time=0.0, tolerance=1.0)
    initial_route_start = guidance.route_start
    assert initial_route_start is not None

    leader.state = AircraftState(vec2(-100.0, 80.0), np.pi)
    guidance.preferred_velocity(leader, agents, time=1.0, tolerance=1.0)

    assert guidance.route_start is not None
    np.testing.assert_array_equal(
        guidance.route_start.position,
        leader.state.position,
    )
    assert not np.array_equal(
        guidance.route_start.position,
        initial_route_start.position,
    )


def test_formation_guidance_tracks_rotated_slot_until_leader_completion():
    params = AircraftParams(
        speed=5.0,
        max_bank_angle=float(np.deg2rad(30.0)),
        radius=1.0,
    )
    goal_state = AircraftState(vec2(50.0, 20.0), np.pi / 4.0)
    leader = Agent(
        id="L",
        params=params,
        state=AircraftState(vec2(0.0, 0.0), 0.0),
        goal=goal_state.position.copy(),
    )
    follower = Agent(
        id="F",
        params=params,
        state=AircraftState(vec2(-25.0, 15.0), 0.0),
        goal=goal_state.position.copy(),
    )
    slot = vec2(-15.0, 10.0)
    initial_slot_error = np.linalg.norm(
        follower.state.position - (leader.state.position + slot)
    )
    guidance = FormationGuidance(
        LeaderPathGuidance("L", goal_state),
        slots={"F": slot},
        position_gain=0.5,
    )
    world = World(
        agents=[leader, follower],
        planner=PreferredVelocityPlanner(),
        guidance=guidance,
        dt=0.05,
        goal_tolerance=1.0,
    )

    world.run(400)

    c = np.cos(leader.state.heading)
    s = np.sin(leader.state.heading)
    rotation = np.array([[c, -s], [s, c]])
    slot_position = leader.state.position + rotation @ slot
    assert world.all_arrived()
    assert (
        np.linalg.norm(follower.state.position - slot_position)
        < initial_slot_error
    )
