"""Dubins paths, control primitives and the Dubins-reachable velocity set.

Two things live here, both ALGORITHM STUBS:

1. Classic Dubins-path machinery (shortest constant-curvature path between two
   oriented configurations), useful for the projection-style baseline.
2. The discrete *control-primitive* generator used by the practical planner:
   a small fan of maneuvers (max-left, moderate-left, straight, moderate-right,
   max-right) propagated over a short horizon.

Also declared here is the notion of the *Dubins-reachable velocity set*
``V_Dubins-reachable`` over a horizon ``T_h`` — the arc of headings the aircraft
can actually reach given its turn-rate limit — which the kinodynamic ORCA
planner intersects with ``V_ORCA-safe``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import AircraftParams, AircraftState


# --------------------------------------------------------------------------- #
# Discrete control primitives (practical planner)
# --------------------------------------------------------------------------- #
@dataclass
class ControlPrimitive:
    """A short, fixed maneuver the aircraft can commit to for one horizon.

    Attributes
    ----------
    name:
        Human-readable label, e.g. ``"max_left"``, ``"straight"``.
    turn_rate:
        Commanded (constant) yaw rate for the primitive (rad/s).
    """

    name: str
    turn_rate: float


@dataclass
class PropagatedPrimitive:
    """A control primitive together with its predicted rollout."""

    primitive: ControlPrimitive
    states: list[AircraftState]
    end_velocity: np.ndarray


def generate_primitives(params: AircraftParams, n: int = 5) -> list[ControlPrimitive]:
    """Generate a symmetric fan of turn-rate primitives within the turn limit.

    Intended default is ``[max_left, moderate_left, straight, moderate_right,
    max_right]``. Should respect ``max_turn_rate(params)``.

    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "generate_primitives: build the discrete maneuver fan — not implemented yet."
    )


def propagate_primitive(
    state: AircraftState,
    primitive: ControlPrimitive,
    params: AircraftParams,
    horizon: float,
    dt: float,
) -> PropagatedPrimitive:
    """Roll a primitive forward over ``horizon`` and return the trajectory.

    ALGORITHM STUB — not implemented yet. (Note: forward kinematics itself is
    available in :mod:`orca_dubins.dynamics`; this wrapper collects the rollout
    and end velocity for evaluation against ORCA constraints.)
    """
    raise NotImplementedError(
        "propagate_primitive: roll out a primitive over the horizon — not implemented yet."
    )


# --------------------------------------------------------------------------- #
# Dubins-reachable velocity set (kinodynamic ORCA)
# --------------------------------------------------------------------------- #
def dubins_reachable_velocities(
    state: AircraftState,
    params: AircraftParams,
    horizon: float,
) -> object:
    """Describe ``V_Dubins-reachable`` over ``T_h = horizon``.

    Conceptually the arc of velocity vectors (constant speed, heading within the
    reachable turn range) attainable within the horizon. Return type is left
    open (e.g. a heading interval, a sampled arc, or a polygon) for you to
    decide during prototyping.

    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "dubins_reachable_velocities: compute the reachable heading arc — not implemented yet."
    )


# --------------------------------------------------------------------------- #
# Classic Dubins shortest path (projection baseline)
# --------------------------------------------------------------------------- #
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
