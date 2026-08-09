"""
Dubins paths, control primitives, and Dubins-reachable velocity set.

We experiment with two separate approaches here:
1. Discrete control primitives for first-pass validation
2. Real Dubins paths for real, continuous optimization

For the second method, we declare the Dubins-reachable velocity set over a time horizon: the 
arc of headings the aircraft can actually reach given its turn rate limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import AircraftParams, AircraftState
from ..dynamics import integrate, max_turn_rate


# -------------------- Discrete control primitives --------------------
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


# ------------------ Dubins-reachable velocity set ------------------
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


# ------------------   Classic Dubins shortest path ------------------ 
@dataclass
class DubinsPath:
    """Shortest constant-curvature path between two oriented configurations."""

    length: float
    # e.g. segment types ("LSL", "RSR", ...) and segment parameters
    segments: object


def dubins_shortest_path(
    start: AircraftState,
    goal: AircraftState,
    turn_radius: float,
) -> DubinsPath:
    """Compute the shortest Dubins path between two configurations.

    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "dubins_shortest_path: classic Dubins path computation — not implemented yet."
    )
