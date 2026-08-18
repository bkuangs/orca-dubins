"""Classical shortest Dubins paths and exact arc-length sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from ..dynamics import integrate
from ..types import AircraftState

PathType = Literal["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]
SegmentParameters = tuple[float, float, float]


@dataclass(frozen=True)
class DubinsPath:
    """Shortest constant-curvature path between two oriented configurations."""

    path_type: PathType
    parameters: SegmentParameters
    turn_radius: float

    @property
    def segment_lengths(self) -> tuple[float, float, float]:
        return tuple(
            self.turn_radius * parameter
            for parameter in self.parameters
        )

    @property
    def length(self) -> float:
        return sum(self.segment_lengths)


def sample_dubins_path(
    start: AircraftState,
    path: DubinsPath,
    distance: float,
) -> AircraftState:
    """Return the pose reached at an arc-length distance along a Dubins path."""
    if not np.isfinite(distance):
        raise ValueError("distance must be finite")
    if distance < 0.0 or distance > path.length:
        raise ValueError("distance must lie within the path")

    state = AircraftState(
        position=start.position.copy(),
        heading=start.heading,
    )
    remaining = distance
    turn_rates = {
        "L": 1.0 / path.turn_radius,
        "S": 0.0,
        "R": -1.0 / path.turn_radius,
    }

    for segment, segment_length in zip(
        path.path_type,
        path.segment_lengths,
        strict=True,
    ):
        if remaining <= 0.0:
            break
        travel = min(remaining, segment_length)
        state = integrate(
            state,
            turn_rate=turn_rates[segment],
            speed=1.0,
            dt=travel,
        )
        remaining -= travel

    return state


def _wrap_angle(angle: float) -> float:
    return angle % (2.0 * np.pi)


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        c * vector[0] - s * vector[1],
        s * vector[0] + c * vector[1],
    ])


def _lsl(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters:
    first_center = start.position + turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    second_center = goal.position + turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )
    center_delta = second_center - first_center
    straight_length = float(np.linalg.norm(center_delta))
    straight_heading = float(np.arctan2(center_delta[1], center_delta[0]))

    return (
        _wrap_angle(straight_heading - start.heading),
        straight_length / turn_radius,
        _wrap_angle(goal.heading - straight_heading),
    )


def _rsr(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters:
    first_center = start.position - turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    second_center = goal.position - turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )
    center_delta = second_center - first_center
    straight_length = float(np.linalg.norm(center_delta))
    straight_heading = float(np.arctan2(center_delta[1], center_delta[0]))

    return (
        _wrap_angle(start.heading - straight_heading),
        straight_length / turn_radius,
        _wrap_angle(straight_heading - goal.heading),
    )


def _lsr(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    first_center = start.position + turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    second_center = goal.position - turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )
    center_delta = second_center - first_center
    center_distance = float(np.linalg.norm(center_delta))
    if center_distance < 2.0 * turn_radius:
        return None

    center_direction = center_delta / center_distance
    tangent_angle = float(np.arcsin(2.0 * turn_radius / center_distance))
    straight_length = float(np.sqrt(max(
        0.0,
        center_distance**2 - (2.0 * turn_radius) ** 2,
    )))
    straight_direction = _rotate(center_direction, tangent_angle)
    straight_heading = float(
        np.arctan2(straight_direction[1], straight_direction[0])
    )

    return (
        _wrap_angle(straight_heading - start.heading),
        straight_length / turn_radius,
        _wrap_angle(straight_heading - goal.heading),
    )


def _rsl(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    first_center = start.position - turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    second_center = goal.position + turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )
    center_delta = second_center - first_center
    center_distance = float(np.linalg.norm(center_delta))
    if center_distance < 2.0 * turn_radius:
        return None

    center_direction = center_delta / center_distance
    tangent_angle = float(np.arcsin(2.0 * turn_radius / center_distance))
    straight_length = float(np.sqrt(max(
        0.0,
        center_distance**2 - (2.0 * turn_radius) ** 2,
    )))
    straight_direction = _rotate(center_direction, -tangent_angle)
    straight_heading = float(
        np.arctan2(straight_direction[1], straight_direction[0])
    )

    return (
        _wrap_angle(start.heading - straight_heading),
        straight_length / turn_radius,
        _wrap_angle(goal.heading - straight_heading),
    )


def _ccc_middle_circle_center(
    first_center: np.ndarray,
    second_center: np.ndarray,
    turn_radius: float,
    side: float,
) -> np.ndarray | None:
    center_delta = second_center - first_center
    center_distance = float(np.linalg.norm(center_delta))
    if center_distance < 1e-12 or center_distance > 4.0 * turn_radius + 1e-12:
        return None

    midpoint = (first_center + second_center) / 2.0
    perpendicular = np.array([-center_delta[1], center_delta[0]]) / center_distance
    height = float(np.sqrt(max(
        0.0,
        (2.0 * turn_radius) ** 2 - (center_distance / 2.0) ** 2,
    )))
    return midpoint + side * height * perpendicular


def _ccc(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
    turn_direction: Literal[-1, 1],
) -> SegmentParameters | None:
    first_center = start.position + turn_direction * turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    second_center = goal.position + turn_direction * turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )
    middle_center = _ccc_middle_circle_center(
        first_center,
        second_center,
        turn_radius,
        side=float(turn_direction),
    )
    if middle_center is None:
        return None

    first_tangent = (first_center + middle_center) / 2.0
    second_tangent = (second_center + middle_center) / 2.0
    first_tangent_heading = (
        float(np.arctan2(
            first_tangent[1] - first_center[1],
            first_tangent[0] - first_center[0],
        ))
        + turn_direction * np.pi / 2.0
    )
    second_tangent_heading = (
        float(np.arctan2(
            second_tangent[1] - second_center[1],
            second_tangent[0] - second_center[0],
        ))
        + turn_direction * np.pi / 2.0
    )

    return (
        _wrap_angle(turn_direction * (first_tangent_heading - start.heading)),
        _wrap_angle(
            -turn_direction * (second_tangent_heading - first_tangent_heading)
        ),
        _wrap_angle(turn_direction * (goal.heading - second_tangent_heading)),
    )


def _rlr(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    return _ccc(start, goal, turn_radius, turn_direction=-1)


def _lrl(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    return _ccc(start, goal, turn_radius, turn_direction=1)


DubinsSolver = Callable[
    [AircraftState, AircraftState, float],
    SegmentParameters | None,
]

_SOLVERS: dict[PathType, DubinsSolver] = {
    "LSL": _lsl,
    "RSR": _rsr,
    "LSR": _lsr,
    "RSL": _rsl,
    "RLR": _rlr,
    "LRL": _lrl,
}


def dubins_shortest_path(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> DubinsPath:
    """Compute the shortest Dubins path between two configurations."""
    if turn_radius <= 0.0:
        raise ValueError("turn_radius must be positive")
    if not np.all(np.isfinite(start.position)):
        raise ValueError("start position must be finite")
    if not np.all(np.isfinite(goal.position)):
        raise ValueError("goal position must be finite")
    if not np.isfinite(start.heading) or not np.isfinite(goal.heading):
        raise ValueError("headings must be finite")

    same_position = np.allclose(start.position, goal.position)
    same_heading = np.isclose(
        _wrap_angle(goal.heading - start.heading),
        0.0,
        atol=1e-12,
    )
    if same_position and same_heading:
        return DubinsPath(
            path_type="LSL",
            parameters=(0.0, 0.0, 0.0),
            turn_radius=turn_radius,
        )

    paths: list[DubinsPath] = []
    for path_type, solve in _SOLVERS.items():
        parameters = solve(start, goal, turn_radius)
        if parameters is not None:
            paths.append(
                DubinsPath(
                    path_type=path_type,
                    parameters=parameters,
                    turn_radius=turn_radius,
                )
            )

    if not paths:
        raise RuntimeError("No feasible Dubins path found")
    return min(paths, key=lambda path: path.length)
