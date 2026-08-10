"""
Kinodynamically-constrained ORCA planner.

Restrict ORCA to the velocities a fixed-wing aircraft can actually 
reach over a short horizon given its turn-rate limit. Then, choose 
the safest reachable velocity:

This preserves ORCA's decentralized, fast local avoidance without needing the
aircraft to be holonomic.
"""

from __future__ import annotations

import numpy as np

from ..types import Agent
from .base import AvoidancePlanner
from ..dubins import dubins_reachable_velocities, wrap_angle, max_turn_rate, ReachableVelocityArc
from ..orca import orca_safe_velocities, satisfies_half_planes


class ReachableOrcaPlanner(AvoidancePlanner):
    """
    ORCA optimization over the Dubins-reachable velocity set.
    """

    name = "reachable_orca"

    def __init__(self, responsibility: float = 0.5) -> None:
        #: Reciprocal avoidance share (0.5 = symmetric).
        self.responsibility = responsibility

    def compute_velocity(
        self,
        ego: Agent,
        neighbours: list[Agent],
        v_pref: np.ndarray,
        horizon: float,
        dt: float,
    ) -> np.ndarray:
        """
        Sampled first version, not yet an exact continuous optimizer.

        TODO: An exact continuous optimizer can later find feasible angular intervals analytically.
        """
        if horizon < dt:
            raise ValueError("horizon must be at least one control timestep")

        half_planes = orca_safe_velocities(
            ego,
            neighbours,
            horizon,
            self.responsibility,
            time_step=dt,
        )
        reachable = dubins_reachable_velocities(
            ego.state,
            ego.params,
            horizon,
        )

        """
        This is the "optimization" step; since speed is fixed, we only need to optimize one variable: heading. 
        We parameterize reachable velocities by a "heading offset" delta; that is, changes in heading will 
        be the only input variable that changes the terminal velocity output. Among these velocities that
        satisfy ORCA, we want to choose the closest to our preferred velocity. Since all candidates have the
        same speed, we are essentially just choosing the heading most aligned with v_pref.
        
        For a first implementation, we will sample 181 headings across the arc.
        
        Note that the chosen velocity is the desired heading at the END of the horizon; since we most likely 
        cannot directly match that at the next simulation step, we must return a velocity that turns towards it
        as much as it can.
        """
        offsets = np.linspace(
            -reachable.half_angle,
            reachable.half_angle,
            181,
        )
        headings = reachable.center_heading + offsets
        velocities = reachable.speed * np.column_stack(
            (np.cos(headings), np.sin(headings))
        )

        feasible = [
            velocity
            for velocity in velocities
            if satisfies_half_planes(velocity, half_planes)
        ]

        if feasible:
            selected = min(
                feasible,
                key=lambda velocity: np.linalg.norm(velocity - v_pref),
            )

        # We use the same heuristic from control primitives if no velocity commands can escape collision
        # Just find the velocity that LEAST VIOLATES the half plane
        else:
            selected = min(
                velocities,
                key=lambda velocity: (
                    self._aggregate_violation(velocity, half_planes),
                    np.linalg.norm(velocity - v_pref),
                ),
            )

        # Re: above, we must return a command to execute NOW, even if we have the desired velocity at 
        # then end of the trajectory
        selected_heading = float(np.arctan2(selected[1], selected[0]))
        heading_error = wrap_angle(selected_heading - ego.state.heading)
        step_error = float(
            np.clip(
                heading_error,
                -max_turn_rate(ego.params) * dt,
                max_turn_rate(ego.params) * dt,
            )
        )
        commanded_heading = ego.state.heading + step_error

        return ego.params.speed * np.array(
            [np.cos(commanded_heading), np.sin(commanded_heading)]
        )