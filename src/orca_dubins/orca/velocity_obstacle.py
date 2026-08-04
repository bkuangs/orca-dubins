"""Velocity Obstacle (VO) construction — ALGORITHM STUB.

A velocity obstacle ``VO_{A|B}`` is the set of relative velocities of A w.r.t. B
that lead to a collision within some time horizon. It is the geometric
primitive underneath ORCA's reciprocal half-planes.

Nothing here is implemented yet — fill these in when prototyping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityObstacle:
    """Truncated velocity obstacle between two disc agents.

    Suggested (not enforced) fields to populate when implementing:
    ``apex``, ``left_leg``, ``right_leg`` direction vectors, and the
    truncation arc defined by the combined radius and time horizon.
    """

    apex: np.ndarray
    left_leg: np.ndarray
    right_leg: np.ndarray


def velocity_obstacle(
    rel_position: np.ndarray,
    combined_radius: float,
    time_horizon: float,
) -> VelocityObstacle:
    """Construct the (truncated) velocity obstacle for one neighbour.

    Parameters
    ----------
    rel_position:
        Position of the neighbour relative to ego (``p_B - p_A``).
    combined_radius:
        ``r_A + r_B``.
    time_horizon:
        Planning horizon ``tau`` over which collisions are considered.

    Returns
    -------
    VelocityObstacle

    Notes
    -----
    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "velocity_obstacle: construct the truncated VO cone/disc — not implemented yet."
    )


def in_velocity_obstacle(rel_velocity: np.ndarray, vo: VelocityObstacle) -> bool:
    """Test whether a relative velocity lies inside a velocity obstacle.

    ALGORITHM STUB — not implemented yet.
    """
    raise NotImplementedError(
        "in_velocity_obstacle: point-in-VO test — not implemented yet."
    )
