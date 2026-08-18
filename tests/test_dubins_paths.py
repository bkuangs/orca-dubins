"""Geometric tests for classical Dubins paths."""

from __future__ import annotations

import numpy as np
import pytest

from orca_dubins.dubins import DubinsPath, dubins_shortest_path, sample_dubins_path
from orca_dubins.dynamics import wrap_angle
from orca_dubins.types import AircraftState, vec2


@pytest.mark.parametrize(
    ("goal", "expected_path_type"),
    [
        (AircraftState(vec2(-4.0, -2.0), 0.0), "LSL"),
        (AircraftState(vec2(-4.0, -4.0), -np.pi), "RSR"),
        (AircraftState(vec2(-4.0, 0.0), -np.pi), "LSR"),
        (AircraftState(vec2(-4.0, -4.0), -3.0 * np.pi / 4.0), "RSL"),
        (AircraftState(vec2(-2.5, 1.0), 3.0 * np.pi / 4.0), "RLR"),
        (AircraftState(vec2(-2.5, -2.5), -3.0 * np.pi / 8.0), "LRL"),
    ],
)
def test_shortest_path_sampling_reaches_goal(
    goal: AircraftState,
    expected_path_type: str,
):
    start = AircraftState(vec2(0.0, 0.0), 0.0)

    path = dubins_shortest_path(start, goal, turn_radius=1.0)
    end = sample_dubins_path(start, path, path.length)

    assert path.path_type == expected_path_type
    if path.path_type in ("RLR", "LRL"):
        assert path.parameters[1] > np.pi
    np.testing.assert_allclose(end.position, goal.position, atol=1e-12)
    assert wrap_angle(end.heading - goal.heading) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("distance", "expected_position", "expected_heading"),
    [
        (0.0, vec2(0.0, 0.0), 0.0),
        (np.pi / 2.0, vec2(1.0, 1.0), np.pi / 2.0),
        (np.pi / 2.0 + 2.0, vec2(1.0, 3.0), np.pi / 2.0),
        (np.pi + 2.0, vec2(0.0, 4.0), np.pi),
    ],
)
def test_sampling_at_segment_boundaries(
    distance: float,
    expected_position: np.ndarray,
    expected_heading: float,
):
    start = AircraftState(vec2(0.0, 0.0), 0.0)
    path = DubinsPath(
        path_type="LSL",
        parameters=(np.pi / 2.0, 2.0, np.pi / 2.0),
        turn_radius=1.0,
    )

    sample = sample_dubins_path(start, path, distance)

    np.testing.assert_allclose(sample.position, expected_position, atol=1e-12)
    assert wrap_angle(sample.heading - expected_heading) == pytest.approx(
        0.0,
        abs=1e-12,
    )


@pytest.mark.parametrize("distance", [-1.0, np.nan, np.inf])
def test_sampling_rejects_invalid_distance(distance: float):
    start = AircraftState(vec2(0.0, 0.0), 0.0)
    path = DubinsPath(
        path_type="LSL",
        parameters=(0.0, 1.0, 0.0),
        turn_radius=1.0,
    )

    with pytest.raises(ValueError, match="distance"):
        sample_dubins_path(start, path, distance)


def test_sampling_rejects_distance_past_path_end():
    start = AircraftState(vec2(0.0, 0.0), 0.0)
    path = DubinsPath(
        path_type="LSL",
        parameters=(0.0, 1.0, 0.0),
        turn_radius=1.0,
    )

    with pytest.raises(ValueError, match="distance"):
        sample_dubins_path(start, path, path.length + 1.0)
