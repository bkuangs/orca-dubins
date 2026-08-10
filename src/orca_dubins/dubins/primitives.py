"""
Dubins paths, control primitives, and Dubins-reachable velocity set.

We experiment with two approaches here:
1. Discrete control primitives for first-pass validation
2. Real Dubins paths for real, continuous optimization

For the second method, we declare the Dubins-reachable velocity set over a time horizon: the 
arc of headings the aircraft can actually reach given its turn rate limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Callable

import numpy as np

from ..types import AircraftParams, AircraftState
from ..dynamics import integrate, max_turn_rate


# ---------------- METHOD 1: Discrete control primitives ----------------
@dataclass
class ControlPrimitive:
    """
    A short, fixed control command over one time horizon -> "What maneuver should the aircraft apply?"
    """

    name: str
    turn_rate: float


@dataclass
class PropagatedPrimitive:
    """
    A control command and the predicted result of executing it -> "What will happen after this primitive is applied?"

    ControlPrimitive
        "moderate_left", +ω/2
                │
                │ propagate over horizon
                ▼
    PropagatedPrimitive
        primitive: moderate_left
        states:    [state₀, state₁, ..., state_final]
        end_velocity: v_final
    """

    primitive: ControlPrimitive
    states: list[AircraftState]
    end_velocity: np.ndarray


def generate_primitives(params: AircraftParams, n: int = 9) -> list[ControlPrimitive]:
    """
    Generate a symmetric fan of turn-rate primitives within the turn limit.

    "Symmetric fan" gives the aircraft several dynamic possible ways to turn, with
    matching left and right commands.

    More primitives: better coverage, more computation.
    Fewer primitives: faster, but may miss a narrow safe maneuver.  

    Importantly, there should be an odd number of primitives. We will start with 5.
    """
    # Positive rate is left; negative rate is right.
    if n < 3:
        raise ValueError("n must be at least 3")
    if n % 2 == 0:
        raise ValueError("n must be odd so the fan includes straight flight")
    if params.speed <= 0:
        raise ValueError("speed must be positive")

    rate_limit = float(max_turn_rate(params))

    turn_rates = np.linspace(rate_limit, -rate_limit, n)
    middle = n // 2

    primitives = []
    for index, turn_rate in enumerate(turn_rates):
        if index == 0:
            name = "max_left"
        elif index == n - 1:
            name = "max_right"
        elif index == middle:
            name = "straight"
        elif index < middle:
            name = f"left_{index}"
        else:
            name = f"right_{n - 1 - index}"

        primitives.append(
            ControlPrimitive(
                name=name,
                turn_rate=float(turn_rate),
            )
        )

    return primitives


def propagate_primitive(
    state: AircraftState,
    primitive: ControlPrimitive,
    params: AircraftParams,
    horizon: float,
    dt: float,
) -> PropagatedPrimitive:
    """
    Repeatedly apply one constant turn rate over the horizon.

    This function will repeatedly evolve (position, heading), which define an AircraftState. 

    States update according to two equations:
    - psi_dot = turn_rate (of primitive)
    - p_dot   = v * Vec2(cos(psi), sin(psi)) where v is velocity

    We record each intermediate state and compute the terminal velocity from the final heading.

    The resulting states array forms a curved trajectory.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if params.speed <= 0:
        raise ValueError("speed must be positive")

    rate_limit = float(max_turn_rate(params))
    if abs(primitive.turn_rate) > rate_limit + 1e-12:
        raise ValueError("primitive turn rate exceeds aircraft limit")

    current = AircraftState(
        position=state.position.copy(),     # don't mutate original
        heading=state.heading
    )
    states = [current]
    elapsed = 0.0

    while elapsed < horizon:
        step = min(dt, horizon - elapsed)   # prevent under/overshooting the time horizon

        current = integrate(
            state=current,
            turn_rate=primitive.turn_rate,
            speed=params.speed,
            dt=step
        )

        states.append(current)
        elapsed += step

    end_velocity = current.velocity(params.speed)

    return PropagatedPrimitive(
        primitive=primitive,
        states=states,
        end_velocity=end_velocity
    )


# -------------- METHOD 2: Real Dubins with continuous optimization --------------
def dubins_reachable_velocities(
    state: AircraftState,
    params: AircraftParams,
    horizon: float,
) -> object:
    """
    Describe ``V_Dubins-reachable`` over ``T_h = horizon``.

    Conceptually the arc of velocity vectors (constant speed, heading within the
    reachable turn range) attainable within the horizon. Return type is left
    open (e.g. a heading interval, a sampled arc, or a polygon) for you to
    decide during prototyping.

    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "dubins_reachable_velocities: compute the reachable heading arc — not implemented yet."
    )


PathType = Literal["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]
SegmentParameters = tuple[float, float, float]  # angle, straight distance, angle
@dataclass(frozen=True)
class DubinsPath:
    """
    Dubins path: the shortest constant-curvature path between two oriented configurations.
    """

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
    

def _wrap_angle(angle: float) -> float:
    return angle % (2.0 * np.pi)


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    """
    Helper to rotate a vector by an arbitrary angle
    """
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
) -> SegmentParameters | None:
    pc1 = start.position + turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    pc2 = goal.position + turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )

    d = pc2 - pc1
    s = float(np.linalg.norm(d))
    theta = float(np.arctan2(d[1], d[0]))               # angle between S and x-axis

    first_turn = _wrap_angle(theta - start.heading)     # turn lengths must be nonnegative angles in the right direction
    final_turn = _wrap_angle(goal.heading - theta)

    return first_turn, s / turn_radius, final_turn


def _rsr(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    pc1 = start.position - turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    pc2 = goal.position - turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )

    d = pc2 - pc1
    s = float(np.linalg.norm(d))
    theta = float(np.arctan2(d[1], d[0]))

    first_turn = _wrap_angle(start.heading - theta)
    final_turn = _wrap_angle(theta - goal.heading)

    return first_turn, s / turn_radius, final_turn


def _lsr(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    pc1 = start.position + turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    pc2 = goal.position - turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )

    d = pc2 - pc1
    s = float(np.linalg.norm(d)) 
    if s < 2.0 * turn_radius: return None

    u = d / s
    alpha = float(np.arcsin(2.0 * turn_radius / s))

    l = float(np.sqrt(max(
        0.0,
        s**2 - (2.0 * turn_radius) ** 2,
    )))
    l_dir = _rotate(u, alpha)

    # Convert direction vector to heading angle
    l_heading = float(
        np.arctan2(l_dir[1], l_dir[0])
    )

    first_turn = _wrap_angle(l_heading - start.heading)
    final_turn = _wrap_angle(l_heading - goal.heading)

    return first_turn, l / turn_radius, final_turn


def _rsl(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    pc1 = start.position - turn_radius * np.array(
        [-np.sin(start.heading), np.cos(start.heading)]
    )
    pc2 = goal.position + turn_radius * np.array(
        [-np.sin(goal.heading), np.cos(goal.heading)]
    )

    d = pc2 - pc1
    s = float(np.linalg.norm(d)) 
    if s < 2.0 * turn_radius: return None

    u = d / s
    alpha = float(np.arcsin(2.0 * turn_radius / s))

    l = float(np.sqrt(max(
        0.0,
        s**2 - (2.0 * turn_radius) ** 2,
    )))
    l_dir = _rotate(u, -alpha)

    # Convert direction vector to heading angle
    l_heading = float(
        np.arctan2(l_dir[1], l_dir[0])
    )

    first_turn = _wrap_angle(start.heading - l_heading)
    final_turn = _wrap_angle(goal.heading - l_heading)

    return first_turn, l / turn_radius, final_turn


def _rlr(start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    return None


def _lrl(start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> SegmentParameters | None:
    return None


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
    """
    Compute the shortest Dubins path between two configurations.
    """
    # Error handling and input validation
    if turn_radius <= 0.0: raise ValueError("turn_radius must be positive")
    if not np.all(np.isfinite(start.position)): raise ValueError("start position must be finite")
    if not np.all(np.isfinite(goal.position)): raise ValueError("goal position must be finite")
    if not np.isfinite(start.heading) or not np.isfinite(goal.heading): raise ValueError("headings must be finite")
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

    # Select Dubins shortest path
    paths: list[DubinsPath] = []

    for path_type, solve in _SOLVERS.items():
        parameters = solve(start, goal, turn_radius)

        if parameters is None:
            continue

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
