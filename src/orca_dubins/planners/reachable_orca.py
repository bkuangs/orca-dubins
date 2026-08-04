"""Kinodynamically-constrained ORCA planner — ALGORITHM STUB.

Idea
----
Rather than letting ORCA consider the entire velocity circle, restrict it to the
velocities the fixed-wing aircraft can actually reach over a short horizon
``T_h`` given its turn-rate limit. Then choose the safest reachable velocity:

    v* = argmin_v || v - v_pref ||
    s.t. v in V_ORCA-safe          (collision-avoidance half-planes)
         v in V_Dubins-reachable   (kinodynamic reachable arc over T_h)

This preserves ORCA's decentralised, fast local avoidance without pretending the
aircraft is holonomic, and avoids the failure mode where a naive ORCA -> Dubins
projection invalidates ORCA's safety guarantee.
"""

from __future__ import annotations

import numpy as np

from ..types import Agent
from .base import AvoidancePlanner


class ReachableOrcaPlanner(AvoidancePlanner):
    """ORCA optimisation over the Dubins-reachable velocity set.

    ALGORITHM STUB — ``compute_velocity`` is not implemented yet.
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
        # Sketch of the intended pipeline (all pieces are stubs today):
        #   half_planes = orca.orca_safe_velocities(ego, neighbours, horizon, self.responsibility)
        #   reachable   = dubins.dubins_reachable_velocities(ego.state, ego.params, horizon)
        #   v_star      = <solve argmin ||v - v_pref|| over half_planes ∩ reachable>
        #   return v_star
        raise NotImplementedError(
            "ReachableOrcaPlanner.compute_velocity: solve ORCA over the "
            "Dubins-reachable set — not implemented yet."
        )
