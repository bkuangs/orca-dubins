"""
Discrete, control primitive ORCA planner.

We are NOT continuously optimizing over the reachable velocity sets yet. Instead:

1. Generate a small set of Dubins/control primitives (e.g. max-left, moderate-left,
   straight, moderate-right, max-right).
2. Propagate each forward over a short horizon under fixed-wing dynamics.
3. Evaluate each rollout against ORCA's collision constraints.
4. Select the *feasible* maneuver whose resulting velocity is closest to the
   preferred mission velocity.

Every primitive is Dubins-feasible by definition, so the only test is ORCA safety.
"""

from __future__ import annotations

import numpy as np

from ..types import Agent
from .base import AvoidancePlanner
from ..dubins import generate_primitives, propagate_primitive, PropagatedPrimitive
from ..orca import orca_safe_velocities, satisfies_half_planes, HalfPlane


class PrimitiveOrcaPlanner(AvoidancePlanner):
    """
    Select the best ORCA-feasible maneuver from a set of discrete primitives.
    """

    name = "primitive_orca"

    def __init__(self, n_primitives: int = 9, responsibility: float = 0.5) -> None:
        self.n_primitives = n_primitives
        self.responsibility = responsibility

    @staticmethod   # method that doesn't modify itself, so don't need to pass in `self` arg
    def aggregate_violation(
        trajectory: PropagatedPrimitive,
        half_planes: list[HalfPlane]
    ) -> float:
        """
        Return the violation score that least violates the safety constraint.
        For this candidate trajectory, see how far its terminal velocity lies on the wrong side of every half-plane.
        "Aggregate" violation scores by getting the signed distance from the half plane boundary.
        """
        total = 0.0

        for half_plane in half_planes:
            margin = float(
                np.dot(
                    trajectory.end_velocity - half_plane.point,
                    half_plane.normal,
                )
            )
            violation = max(0.0, -margin)
            total += violation**2

        return total

    def compute_velocity(
        self,
        ego: Agent,
        neighbours: list[Agent],
        v_pref: np.ndarray,
        horizon: float,
        dt: float,
    ) -> np.ndarray:
        """
        Given a primary agent (ego), its neighboring agents, and our preferred velocity,
        compute safe velocities to enable collision avoidance!

        A velocity is globally ORCA-safe only if it satisfies ALL neighboring half planes.

        Candidate maneuvers are ranked using their terminal velocities. The returned
        velocity points along the selected primitive after one control timestep, so
        the simulation's velocity tracker executes that primitive's turn rate now.
        """

        if horizon < dt:
            raise ValueError("horizon must be at least one control timestep")

        primitives = generate_primitives(ego.params, self.n_primitives)
        half_planes = orca_safe_velocities(
            ego,
            neighbours,
            horizon,
            self.responsibility,
            time_step=dt,
        )

        trajectories = []
        feasible = []

        for p in primitives:
            trajectory = propagate_primitive(ego.state, p, ego.params, horizon, dt)
            trajectories.append(trajectory)

            if satisfies_half_planes(trajectory.end_velocity, half_planes):
                feasible.append(trajectory)

        if feasible:
            # Goal is to choose the closest primitive to preferred velo;
            # that is, minimize || roll.end_velocity - v_pref ||
            best = min(feasible, key=lambda f: np.linalg.norm(f.end_velocity - v_pref))

        else:
            best = min(
                trajectories,
                key=lambda trajectory: (
                    self.aggregate_violation(trajectory, half_planes),
                    np.linalg.norm(trajectory.end_velocity - v_pref),
                ),
            )

        # The planner interface returns a velocity command. Using the first
        # propagated state makes World.steer_toward_velocity recover the exact
        # constant turn rate selected by this primitive for the current step.
        return best.states[1].velocity(ego.params.speed)
