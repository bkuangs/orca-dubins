"""ORCA / velocity-obstacle collision-avoidance primitives (algorithm stubs)."""

from .orca import (
    HalfPlane,
    orca_half_plane,
    orca_safe_velocities,
    satisfies_half_planes,
)
from .velocity_obstacle import (
    VelocityObstacle,
    in_velocity_obstacle,
    velocity_obstacle,
)

__all__ = [
    "HalfPlane",
    "orca_half_plane",
    "orca_safe_velocities",
    "satisfies_half_planes",
    "VelocityObstacle",
    "in_velocity_obstacle",
    "velocity_obstacle",
]
