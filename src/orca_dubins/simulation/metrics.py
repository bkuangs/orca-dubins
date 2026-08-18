"""Metrics for recorded swarm simulations."""

from __future__ import annotations

import numpy as np

from ..types import Snapshot
from .guidance import FormationGuidance, LeaderPathGuidance


def avoidance_is_active(
    snapshot: Snapshot,
    agent_id: str,
    angle_threshold: float = float(np.deg2rad(2.0)),
) -> bool:
    """Return whether neighbors changed an agent's nominal planner command."""
    if not np.isfinite(angle_threshold) or angle_threshold < 0.0:
        raise ValueError("angle_threshold must be nonnegative and finite")
    commanded = snapshot.commanded.get(agent_id)
    nominal = snapshot.nominal.get(agent_id)
    if commanded is None or nominal is None:
        return False

    scale = float(np.linalg.norm(commanded) * np.linalg.norm(nominal))
    if scale < 1e-12:
        return False
    cosine = float(np.clip(np.dot(commanded, nominal) / scale, -1.0, 1.0))
    return float(np.arccos(cosine)) > angle_threshold


def avoidance_intervention_count(
    history: list[Snapshot],
    angle_threshold: float = float(np.deg2rad(2.0)),
) -> int:
    """Count agent-frames where neighbors changed the planner command."""
    return sum(
        avoidance_is_active(snapshot, agent_id, angle_threshold)
        for snapshot in history
        for agent_id in snapshot.positions
    )


def minimum_pairwise_separation(history: list[Snapshot]) -> float:
    """Return the minimum distance between any two agents over the run."""
    if not history:
        raise ValueError("history must not be empty")
    agent_ids = list(history[0].positions)
    if len(agent_ids) < 2:
        return float("inf")

    return min(
        float(
            np.linalg.norm(
                snapshot.positions[first] - snapshot.positions[second]
            )
        )
        for snapshot in history
        for index, first in enumerate(agent_ids)
        for second in agent_ids[index + 1:]
    )


def leader_cross_track_rmse(
    history: list[Snapshot],
    guidance: LeaderPathGuidance,
    spacing: float = 5.0,
) -> float:
    """Return leader position RMSE relative to the active Dubins route."""
    if not history:
        raise ValueError("history must not be empty")
    route = guidance.route_points(spacing)
    if len(route) == 0:
        raise ValueError("leader route has not been initialized")

    squared_errors = []
    for snapshot in history:
        position = snapshot.positions[guidance.leader_id]
        squared_errors.append(
            min(
                float(np.dot(position - route_point, position - route_point))
                for route_point in route
            )
        )
    return float(np.sqrt(np.mean(squared_errors)))


def assigned_slot_rmse(
    history: list[Snapshot],
    guidance: FormationGuidance,
    start_index: int = 0,
) -> float:
    """Return follower RMSE relative to their currently assigned slots."""
    if not 0 <= start_index < len(history):
        raise ValueError("start_index must select a recorded snapshot")

    leader_id = guidance.leader_guidance.leader_id
    squared_errors = []
    for snapshot in history[start_index:]:
        for follower_id in guidance.slots:
            slot = guidance.slot_position_from_pose(
                snapshot.positions[leader_id],
                snapshot.headings[leader_id],
                follower_id,
                snapshot.time,
            )
            error = snapshot.positions[follower_id] - slot
            squared_errors.append(float(np.dot(error, error)))

    return float(np.sqrt(np.mean(squared_errors)))
