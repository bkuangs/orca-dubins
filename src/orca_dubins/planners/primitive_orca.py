"""Discrete control-primitive avoidance planner — ALGORITHM STUB.

The practical strategy: don't continuously optimise over the reachable velocity
set. Instead:

1. Generate a small set of Dubins/control primitives (max-left, moderate-left,
   straight, moderate-right, max-right).
2. Propagate each forward over a short horizon ``T_h`` under the fixed-wing
   dynamics.
3. Evaluate each rollout against ORCA's collision constraints
   (``V_ORCA-safe``).
4. Select the *feasible* maneuver whose resulting velocity is closest to the
   preferred mission velocity ``v_pref``.

Every primitive is Dubins-feasible by construction, so feasibility w.r.t.
``V_Dubins-reachable`` is automatic; the only test is ORCA safety.
"""

from __future__ import annotations

import numpy as np

from ..types import Agent
from .base import AvoidancePlanner


class PrimitiveOrcaPlanner(AvoidancePlanner):
    """Select the best ORCA-feasible maneuver from a discrete primitive fan.

    ALGORITHM STUB — ``compute_velocity`` is not implemented yet.
    """

    name = "primitive_orca"

    def __init__(self, n_primitives: int = 5, responsibility: float = 0.5) -> None:
        #: Number of maneuvers in the fan (odd -> includes "straight").
        self.n_primitives = n_primitives
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
        #   prims = dubins.generate_primitives(ego.params, self.n_primitives)
        #   half_planes = orca.orca_safe_velocities(ego, neighbours, horizon, self.responsibility)
        #   feasible = []
        #   for p in prims:
        #       roll = dubins.propagate_primitive(ego.state, p, ego.params, horizon, dt)
        #       if orca.satisfies_half_planes(roll.end_velocity, half_planes):
        #           feasible.append(roll)
        #   pick roll minimising ||roll.end_velocity - v_pref||; fall back if none feasible
        raise NotImplementedError(
            "PrimitiveOrcaPlanner.compute_velocity: evaluate primitives against "
            "ORCA constraints and pick the best feasible one — not implemented yet."
        )
