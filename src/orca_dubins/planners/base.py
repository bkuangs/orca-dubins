"""Common interface for local collision-avoidance planners.

Every planner takes the ego agent, its neighbours, the preferred (mission)
velocity and the horizon/timestep, and returns a single commanded velocity
``v*`` for this control cycle. The simulation then realises ``v*`` as a
feasible fixed-wing turn via :func:`orca_dubins.dynamics.steer_toward_velocity`.

The two research strategies (kinodynamic ORCA over the reachable set, and the
discrete control-primitive planner) implement this interface. Both share the
objective

    v* = argmin_v || v - v_pref ||   s.t.   v in V_ORCA-safe  and  v in V_Dubins-reachable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Agent


class AvoidancePlanner(ABC):
    """Abstract base class for a per-agent local avoidance planner."""

    #: Short identifier used in logs/plots.
    name: str = "planner"

    @abstractmethod
    def compute_velocity(
        self,
        ego: Agent,
        neighbours: list[Agent],
        v_pref: np.ndarray,
        horizon: float,
        dt: float,
    ) -> np.ndarray:
        """Return the commanded velocity ``v*`` for ego this cycle.

        Parameters
        ----------
        ego:
            The aircraft being planned for.
        neighbours:
            Other aircraft ego must avoid (already filtered to those relevant,
            e.g. within sensing range — filtering policy is up to the caller).
        v_pref:
            Preferred mission velocity ``v_pref`` (shape ``(2,)``).
        horizon:
            Planning horizon ``T_h`` (s).
        dt:
            Control timestep (s).

        Returns
        -------
        numpy.ndarray
            The chosen velocity vector, shape ``(2,)``.
        """
        raise NotImplementedError
