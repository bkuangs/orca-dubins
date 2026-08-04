"""Runnable, algorithm-free baseline planners.

These exist so the simulation + visualisation pipeline runs end-to-end before
any avoidance algorithm is implemented. They perform NO collision avoidance.
"""

from __future__ import annotations

import numpy as np

from ..types import Agent
from .base import AvoidancePlanner


class PreferredVelocityPlanner(AvoidancePlanner):
    """Ignores neighbours and always commands the preferred (mission) velocity.

    Useful as a sanity check for the harness and as a "no avoidance" control
    baseline to compare the real planners against.
    """

    name = "preferred"

    def compute_velocity(
        self,
        ego: Agent,
        neighbours: list[Agent],
        v_pref: np.ndarray,
        horizon: float,
        dt: float,
    ) -> np.ndarray:
        return np.asarray(v_pref, dtype=float)
