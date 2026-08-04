"""ORCA half-planes and the safe-velocity set — ALGORITHM STUBS.

ORCA (Optimal Reciprocal Collision Avoidance) turns each neighbour's velocity
obstacle into a *half-plane* constraint on the ego velocity. The intersection
of these half-planes is the ORCA-safe velocity set

    V_ORCA-safe = { v : v satisfies every neighbour's ORCA half-plane }.

The planners then choose ``v*`` from this set (intersected with the
Dubins-reachable set). Nothing here is implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Agent


@dataclass
class HalfPlane:
    """A linear velocity constraint ``(v - point) . normal >= 0``.

    ``point`` lies on the boundary line and ``normal`` points into the feasible
    (collision-free) side.
    """

    point: np.ndarray
    normal: np.ndarray


def orca_half_plane(
    ego: Agent,
    neighbour: Agent,
    time_horizon: float,
    responsibility: float = 0.5,
) -> HalfPlane:
    """Build the ORCA half-plane that ego must respect for one neighbour.

    ``responsibility`` splits avoidance effort between the two agents (0.5 for
    reciprocal, symmetric avoidance; 1.0 if the neighbour is treated as a static
    or non-cooperative obstacle).

    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "orca_half_plane: derive the reciprocal half-plane from the VO — not implemented yet."
    )


def orca_safe_velocities(
    ego: Agent,
    neighbours: list[Agent],
    time_horizon: float,
    responsibility: float = 0.5,
) -> list[HalfPlane]:
    """Return the ORCA half-plane constraints defining ``V_ORCA-safe`` for ego.

    The safe set is the intersection of the returned half-planes. Selecting the
    optimal feasible velocity from that intersection (the classic ORCA linear
    program) is the planners' job.

    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "orca_safe_velocities: assemble per-neighbour half-planes — not implemented yet."
    )


def satisfies_half_planes(
    velocity: np.ndarray,
    half_planes: list[HalfPlane],
    tolerance: float = 0.0,
) -> bool:
    """Check whether a velocity satisfies every ORCA half-plane.

    Useful for the control-primitive planner, which tests discrete candidate
    velocities against ``V_ORCA-safe`` rather than solving an LP.

    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "satisfies_half_planes: half-plane membership test — not implemented yet."
    )
