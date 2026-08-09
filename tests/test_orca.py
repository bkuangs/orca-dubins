"""Behavioral tests for ORCA half-plane construction and membership."""

from __future__ import annotations

import numpy as np
import pytest

from orca_dubins.orca.orca import (
    HalfPlane,
    orca_half_plane,
    orca_safe_velocities,
    satisfies_half_planes,
)
from orca_dubins.orca.velocity_obstacle import velocity_obstacle
from orca_dubins.types import Agent, AircraftParams, AircraftState, vec2


def _agent(
    agent_id: str,
    position: np.ndarray,
    velocity: np.ndarray,
    radius: float = 0.5,
) -> Agent:
    """Build an agent whose state and constant speed produce ``velocity``."""
    speed = float(np.linalg.norm(velocity))
    heading = 0.0 if speed == 0.0 else float(np.arctan2(velocity[1], velocity[0]))
    return Agent(
        id=agent_id,
        params=AircraftParams(
            speed=speed,
            max_bank_angle=float(np.deg2rad(30.0)),
            radius=radius,
        ),
        state=AircraftState(position=np.asarray(position, dtype=float), heading=heading),
        goal=np.asarray(position, dtype=float) + vec2(10.0, 0.0),
    )


def _pair(relative_velocity: np.ndarray) -> tuple[Agent, Agent]:
    """Create separated agents with the requested ego-minus-neighbor velocity."""
    ego = _agent("A", vec2(0.0, 0.0), np.asarray(relative_velocity, dtype=float))
    neighbor = _agent("B", vec2(5.0, 0.0), vec2(0.0, 0.0))
    return ego, neighbor


def test_half_plane_accepts_positive_side_and_boundary():
    half_plane = HalfPlane(point=vec2(2.0, 0.0), normal=vec2(1.0, 0.0))

    assert satisfies_half_planes(vec2(3.0, 4.0), [half_plane])
    assert satisfies_half_planes(vec2(2.0, -7.0), [half_plane])


def test_half_plane_rejects_negative_side():
    half_plane = HalfPlane(point=vec2(2.0, 0.0), normal=vec2(1.0, 0.0))

    assert not satisfies_half_planes(vec2(1.0, 0.0), [half_plane])


def test_half_plane_membership_honors_tolerance():
    half_plane = HalfPlane(point=vec2(0.0, 0.0), normal=vec2(1.0, 0.0))
    velocity = vec2(-0.05, 0.0)

    assert satisfies_half_planes(velocity, [half_plane], tolerance=0.1)
    assert not satisfies_half_planes(velocity, [half_plane], tolerance=0.01)


def test_every_velocity_satisfies_an_empty_constraint_set():
    assert satisfies_half_planes(vec2(123.0, -456.0), [])


def test_head_on_velocity_is_rejected_by_its_orca_half_plane():
    ego, neighbor = _pair(vec2(2.0, 0.0))

    half_plane = orca_half_plane(ego, neighbor, time_horizon=5.0)
    ego_velocity = ego.state.velocity(ego.params.speed)

    assert np.linalg.norm(half_plane.normal) == pytest.approx(1.0)
    assert not satisfies_half_planes(ego_velocity, [half_plane])
    assert satisfies_half_planes(half_plane.point, [half_plane])
    assert satisfies_half_planes(half_plane.point + half_plane.normal, [half_plane])


def test_cap_projection_builds_expected_half_plane():
    # The relative velocity (0.9, 0) lies inside the near cap of the VO whose
    # truncation circle has center (1, 0) and radius 0.2.
    ego, neighbor = _pair(vec2(0.9, 0.0))

    half_plane = orca_half_plane(ego, neighbor, time_horizon=5.0)

    np.testing.assert_allclose(half_plane.normal, vec2(-1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(half_plane.point, vec2(0.85, 0.0), atol=1e-12)


@pytest.mark.parametrize(
    ("relative_velocity", "leg_name"),
    [
        (vec2(2.0, 0.1), "left_leg"),
        (vec2(2.0, -0.1), "right_leg"),
    ],
)
def test_leg_projection_uses_the_closest_valid_leg(
    relative_velocity: np.ndarray,
    leg_name: str,
):
    ego, neighbor = _pair(relative_velocity)
    horizon = 5.0
    responsibility = 0.5
    half_plane = orca_half_plane(
        ego,
        neighbor,
        time_horizon=horizon,
        responsibility=responsibility,
    )
    vo = velocity_obstacle(
        neighbor.state.position - ego.state.position,
        ego.params.radius + neighbor.params.radius,
        horizon,
    )
    leg = getattr(vo, leg_name)

    # Recover the selected relative-velocity boundary point from the public
    # half-plane result: point = ego_velocity + responsibility * u.
    ego_velocity = ego.state.velocity(ego.params.speed)
    correction = (half_plane.point - ego_velocity) / responsibility
    boundary_point = relative_velocity + correction
    tangent_distance = np.sqrt(
        np.dot(vo.truncation_center, vo.truncation_center)
        - vo.truncation_radius**2
    )

    assert abs(float(boundary_point[0] * leg[1] - boundary_point[1] * leg[0])) < 1e-12
    assert float(np.dot(boundary_point, leg)) >= tangent_distance - 1e-12
    assert np.linalg.norm(half_plane.normal) == pytest.approx(1.0)


def test_current_velocity_satisfies_constraint_when_already_safe():
    ego, neighbor = _pair(vec2(0.0, 2.0))

    half_plane = orca_half_plane(ego, neighbor, time_horizon=5.0)
    ego_velocity = ego.state.velocity(ego.params.speed)

    assert satisfies_half_planes(ego_velocity, [half_plane])


def test_safe_velocity_constraints_include_one_half_plane_per_neighbor():
    ego = _agent("A", vec2(0.0, 0.0), vec2(0.9, 0.0))
    neighbors = [
        _agent("B", vec2(5.0, 0.0), vec2(0.0, 0.0)),
        _agent("C", vec2(0.0, 5.0), vec2(0.0, 0.0)),
    ]

    half_planes = orca_safe_velocities(ego, neighbors, time_horizon=5.0)

    assert len(half_planes) == len(neighbors)
    assert all(isinstance(half_plane, HalfPlane) for half_plane in half_planes)


def test_safe_velocity_constraints_are_empty_without_neighbors():
    ego = _agent("A", vec2(0.0, 0.0), vec2(1.0, 0.0))

    assert orca_safe_velocities(ego, [], time_horizon=5.0) == []


@pytest.mark.parametrize("time_horizon", [0.0, -1.0])
def test_half_plane_rejects_nonpositive_time_horizon(time_horizon: float):
    ego, neighbor = _pair(vec2(1.0, 0.0))

    with pytest.raises(ValueError, match="time horizon"):
        orca_half_plane(ego, neighbor, time_horizon=time_horizon)


@pytest.mark.parametrize("responsibility", [-0.01, 1.01])
def test_half_plane_rejects_responsibility_outside_unit_interval(
    responsibility: float,
):
    ego, neighbor = _pair(vec2(1.0, 0.0))

    with pytest.raises(ValueError, match="responsibility"):
        orca_half_plane(
            ego,
            neighbor,
            time_horizon=5.0,
            responsibility=responsibility,
        )


@pytest.mark.parametrize("responsibility", [0.0, 1.0])
def test_half_plane_accepts_responsibility_interval_endpoints(
    responsibility: float,
):
    ego, neighbor = _pair(vec2(0.9, 0.0))

    half_plane = orca_half_plane(
        ego,
        neighbor,
        time_horizon=5.0,
        responsibility=responsibility,
    )

    assert isinstance(half_plane, HalfPlane)


def test_overlapping_agents_receive_an_emergency_separation_constraint():
    ego = _agent("A", vec2(0.0, 0.0), vec2(1.0, 0.0), radius=0.5)
    neighbor = _agent("B", vec2(0.75, 0.0), vec2(-1.0, 0.0), radius=0.5)

    half_plane = orca_half_plane(
        ego,
        neighbor,
        time_horizon=5.0,
        responsibility=0.5,
        time_step=0.1,
    )

    assert np.all(np.isfinite(half_plane.point))
    assert np.all(np.isfinite(half_plane.normal))
    assert np.linalg.norm(half_plane.normal) == pytest.approx(1.0)
    assert satisfies_half_planes(half_plane.point + half_plane.normal, [half_plane])
