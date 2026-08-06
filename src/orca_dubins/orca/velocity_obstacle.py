"""
Velocity Obstacle (VO) construction.

A velocity obstacle ``VO_{A|B}`` is the set of relative velocities of A w.r.t. B
that lead to a collision within some time horizon. It is the geometric
primitive underneath ORCA's reciprocal half-planes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPSILON = 1e-12


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class VelocityObstacle:
    """
    Truncated velocity obstacle between two disc agents.
    """

    apex: np.ndarray
    left_leg: np.ndarray
    right_leg: np.ndarray
    truncation_center: np.ndarray | None = None
    truncation_radius: float | None = None
    combined_radius: float | None = None
    time_horizon: float | None = None


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
    Uses the convention ``rel_velocity = v_A - v_B``. With
    ``rel_position = p_B - p_A``, unsafe velocities point roughly toward
    ``rel_position``.
    """
    rel_position = np.asarray(rel_position, dtype=float)

    apex = np.zeros(2, dtype=float)                     # we are in relative velocity space, so ego starts at (0, 0)
    distance = float(np.linalg.norm(rel_position))
    truncation_center = rel_position / time_horizon     # relative velocity for two POINTS to meet at time t
    truncation_radius = combined_radius / time_horizon  # expand the point collision to account for real object size

    if distance <= _EPSILON:                            # objects are at the same position
        left_leg = np.array([0.0, 1.0])
        right_leg = np.array([0.0, -1.0])
    elif combined_radius >= distance:                   # the objects are overlapping
        center_dir = rel_position / distance
        left_leg = np.array([-center_dir[1], center_dir[0]])
        right_leg = -left_leg

    else:
        center_angle = float(np.arctan2(rel_position[1], rel_position[0]))      # bearing from ego to neighbor (quadrant 2)
        half_angle = float(np.arcsin(combined_radius / distance))               # angle from center to either tangent

        left_angle = center_angle + half_angle
        right_angle = center_angle - half_angle

        """
        On a unit circle:

        y
          ^
          |
      •   |  point = (cos θ, sin θ)
       \  |
        \ |
         \|
----------+----------> x
          θ

        Similarly, following give us unit vectors at angle left_angle, right_angle.
        These are the tangent edges around the collision cone
        """
        left_leg = np.array([np.cos(left_angle), np.sin(left_angle)])           # unit vector along circle
        right_leg = np.array([np.cos(right_angle), np.sin(right_angle)])

    return VelocityObstacle(
        apex=apex,
        left_leg=left_leg,
        right_leg=right_leg,
        truncation_center=truncation_center,
        truncation_radius=truncation_radius,
        combined_radius=combined_radius,
        time_horizon=time_horizon,
    )

def in_velocity_obstacle(rel_velocity: np.ndarray, vo: VelocityObstacle) -> bool:
    """
    Test whether a relative velocity lies inside a velocity obstacle.
    """
    if vo.truncation_center is None or vo.truncation_radius is None:
        raise ValueError(""Can't generate a collision cone without center and/or radius"")

    rel_velocity = np.asarray(rel_velocity, dtype=float)
    collision_center_velocity = vo.truncation_center
    collision_radius = vo.truncation_radius

    if float(np.linalg.norm(collision_center_velocity)) <= collision_radius + _EPSILON:      # agents already collide
        return True

    speed = float(np.linalg.norm(rel_velocity))
    if speed <= _EPSILON:
        return False

    direction = rel_velocity / speed

    # Check that right_leg <= direction <= left_leg
    inside_cone = (
        _cross_2d(vo.right_leg, direction) >= -_EPSILON
        and _cross_2d(direction, vo.left_leg) >= -_EPSILON
    )
    if not inside_cone:
        return False

    """
    We can treat collision checking as a ray-circle intersection problem.

    Ray: x(t) = t * d, where t is time horizon and d is the unit distance vector

    We plug this into the circle equation to determine the time t where they intersect:
    - Any point on the circle's border must satisfy the equation: (p - c) dot (p - c) = r^2, where p is the point
    - We substitute the ray equation for the point p: ((td) - c) dot ((td) - c) = r^2

    Expanding this gives us a standard quadratic equation in terms of t
    """
    center_projection = float(np.dot(direction, collision_center_velocity))
    center_distance_sq = float(np.dot(collision_center_velocity, collision_center_velocity))

    # Term under the square root: Negative means ray missed, positive means ray collided
    discriminant = center_projection**2 - (center_distance_sq - collision_radius**2)
    if discriminant < -_EPSILON:
        return False

    # Taking the smaller term gives us the closer intersect (ray passes through two points in circle)
    first_intersection = center_projection - float(np.sqrt(max(0.0, discriminant)))
    return speed + _EPSILON >= first_intersection     # reached OR passed boundary
