"""
ORCA half-planes and the safe-velocity set.

ORCA turns each neighbor's VO into a half plane constraint on the ego velocity. The intersection
of these half-planes is the ORCA-safe velocity set

    V_ORCA-safe = { v : v satisfies every neighbor's ORCA half-plane }.

The planners then choose ``v*`` from this set (intersected with the Dubins-reachable set). 
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Agent
from .velocity_obstacle import velocity_obstacle

_EPSILON = 1e-12

@dataclass
class HalfPlane:
    """
    A linear velocity constraint ``(v - point) . normal >= 0``.

    ``point`` lies on the boundary line and ``normal`` points into the feasible
    (collision-free) side.
    """

    point: np.ndarray
    normal: np.ndarray


def orca_half_plane(
    ego: Agent,
    neighbor: Agent,
    time_horizon: float,
    responsibility: float = 0.5,
) -> HalfPlane:
    """
    Build the ORCA half-plane that ego must respect for one neighbor.

    It's important to understand the difference between the position space and
    velocity space here. The position space is the intuitive understanding of an 
    imminent collision between two objects: a rounded cone, with its tip starting at
    the ego and the base rounded by a circle, with the circle center being the obstacle
    and radius determined by the combined size of both objects.

    On the other hand, the velocity space is a 2D coordinate plane of v_x, v_y which
    represent the x and y magnitudes of the relative velocity between the two objects.
    Each point in the velocity space is the tip of a relative velocity vector. The velocity 
    space is where the "velocity obstacle" is created; that is, the relative velocity (v_x, v_y)
    that would cause a collision (asssuming constant velocity) over a time horizon. 

    The vector `u` is a velocity CHANGE vector that represents the smallest change in velocity 
    needed to reach the boundary of the velocity obstacle (i.e., avoid collision). If there is
    no imminent collision, `u` will still be the shortest vector to the boundary of the VO, which
    helps define the half plane constraint of allowed velocity.

    So why does the edge of the half plane not touch the VO boundary? We would expect the edges 
    to touch at `u`. However, remember that the two objects share responsibility. Each object applies
    only half (by default) of `u`, so the half plane edge is at half the distance to the VO boundary.
    """
    if time_horizon <= 0:
        raise ValueError("No time horizon")
    
    rel_position = neighbor.state.position - ego.state.position
    combined_radius = ego.params.radius + neighbor.params.radius
    vo = velocity_obstacle(rel_position, combined_radius, time_horizon)

    """
    We use vector projections to find the shortest vector `u` between two vectors.
    - Recall that pure dot product returns "how aligned" two vectors are (magnitude)
    - If we multiply this dot product by the vector being projected on, that gets us the tip of the projection (direction)
    - Finally, subtracting the two guarantees the shortest vector between them
    """

    v = ego.state.velocity(ego.params.speed) - neighbor.state.velocity(neighbor.params.speed)     # rel_velocity
    c = vo.truncation_center
    r = vo.truncation_radius

    u_vectors = []

    # Projection onto circle
    w = v - c   # ray from circle center to relative velocity
    w_length = float(np.linalg.norm(w))
    if w_length <= _EPSILON:
        center_length = float(np.linalg.norm(vo.truncation_center))
        direction = -vo.truncation_center / center_length
    else:
        direction = w / w_length    # direction from circle center to relative velocity

    q_cap = c + r * direction       # move a magnitude of circle radius along the direction towards relative velocity
                                    # this lands us right on the circle boundary, towards relative velocity

    u_cap = q_cap - v               # then, the shortest correction vector is simply this boundary point minus relative velocity
    n_cap = direction               # the normal vector wrt the circle at this point is the same as direction

    # We must make sure this point on the circle boundary is along the outer edge and not inside the VO
    point_on_cap = v + u_cap
    cap_is_valid = (
        np.dot(
            point_on_cap,
            point_on_cap - c,
        )
        <= _EPSILON
    )

    if cap_is_valid:
        u_vectors.append(u_cap, n_cap)

    """
    Projecting onto cone legs is much more straightforward, as they are just rays. However,
    we must make sure that the projection point occurs after the tangent point; that is, we must 
    make sure this projection lies after the circular cap

    We can imagine this in the position space: Tangent points occur where the cone legs intersect 
    with the combined radius of b, i.e. where the rays are tangent to the circle. Each cone leg is a unit vector,
    and we are looking for the magnitude along the cone leg that hits the tangent point.

    This is transformed back to velocity space by dividing vectors by time horizon t.
    """

    tangent_distance = np.sqrt(np.linalg.norm(c)**2 - r**2)

    # Left leg projection
    t_left = np.dot(v, vo.left_leg)                      # how aligned rel_velocity and leg are (magnitude)
    q_left = t_left * vo.left_leg                        # magnitude x direction along leg
    u_left = q_left - v                                  # `u` is the shortest vector between leg and rel_velocity
    n_left = np.array([-vo.left_leg[1], vo.left_leg[0]])
    if t_left >= tangent_distance: 
        u_vectors.append(u_left, n_left)

    # Right leg projection
    t_right = np.dot(v, vo.right_leg)
    q_right = t_right * vo.right_leg
    u_right = q_right - v
    n_right = np.array([vo.right_leg[1], -vo.right_leg[0]])
    if t_right >= tangent_distance: 
        u_vectors.append(u_right, n_right)

    # Get the valid candidate with the smallest correction vector
    u, n = min(
        u_vectors,
        key=lambda candidate: np.linalg.norm(candidate[0]),
    )

    # Apply the change in velocity vector to build the half plane
    ego_velocity = ego.state.velocity(ego.params.speed)
    p = ego_velocity + responsibility * u

    return HalfPlane(point=p, normal=n)


def orca_safe_velocities(
    ego: Agent,
    neighbors: list[Agent],
    time_horizon: float,
    responsibility: float = 0.5,
) -> list[HalfPlane]:
    """
    Return the ORCA half-plane constraints defining ``V_ORCA-safe`` for ego.

    The safe set is the intersection of the returned half-planes. Selecting the
    optimal feasible velocity from that intersection (the classic ORCA linear
    program) is the planners' job.
    """
    half_planes = []

    for neighbor in neighbors:
        half_plane = orca_half_plane(ego=ego, neighbor=neighbor, time_horizon=time_horizon, responsibility=responsibility)
        half_planes.append(half_plane)

    return half_planes


def satisfies_half_planes(
    velocity: np.ndarray,
    half_planes: list[HalfPlane],
    tolerance: float = 0.0,
) -> bool:
    """
    Check whether a velocity satisfies every ORCA half-plane.

    Useful for the control-primitive planner, which tests discrete candidate
    velocities against ``V_ORCA-safe`` rather than solving an LP.
    """

