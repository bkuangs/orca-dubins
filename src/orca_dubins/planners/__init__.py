"""Local collision-avoidance planners.

* :class:`AvoidancePlanner` — common interface.
* :class:`PreferredVelocityPlanner` — runnable no-avoidance baseline.
* :class:`ReachableOrcaPlanner` — kinodynamic ORCA over the reachable set (stub).
* :class:`PrimitiveOrcaPlanner` — discrete control-primitive planner (stub).
"""

from .base import AvoidancePlanner
from .baseline import PreferredVelocityPlanner
from .primitive_orca import PrimitiveOrcaPlanner
from .reachable_orca import ReachableOrcaPlanner

__all__ = [
    "AvoidancePlanner",
    "PreferredVelocityPlanner",
    "PrimitiveOrcaPlanner",
    "ReachableOrcaPlanner",
]
