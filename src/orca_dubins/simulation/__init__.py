"""Simulation harness: multi-agent world and scenarios."""

from .guidance import (
    FormationGuidance,
    LeaderPathGuidance,
    MissionGuidance,
    PointGoalGuidance,
    SlotSwap,
)
from .metrics import (
    avoidance_intervention_count,
    avoidance_is_active,
    assigned_slot_rmse,
    leader_cross_track_rmse,
    minimum_pairwise_separation,
)
from .scenario import (
    SCENARIOS,
    crossing,
    formation_goal_state,
    formation_slots,
    formation_slot_swaps,
    head_on,
    swarm_circle,
    swarm_formation,
    swarm_formation_guidance,
    swarm_random,
)
from .world import World

__all__ = [
    "FormationGuidance",
    "LeaderPathGuidance",
    "MissionGuidance",
    "PointGoalGuidance",
    "SlotSwap",
    "avoidance_intervention_count",
    "avoidance_is_active",
    "assigned_slot_rmse",
    "leader_cross_track_rmse",
    "minimum_pairwise_separation",
    "World",
    "SCENARIOS",
    "crossing",
    "formation_goal_state",
    "formation_slots",
    "formation_slot_swaps",
    "head_on",
    "swarm_circle",
    "swarm_formation",
    "swarm_formation_guidance",
    "swarm_random",
]
