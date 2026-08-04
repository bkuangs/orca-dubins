"""Simulation harness: multi-agent world and scenarios."""

from .scenario import SCENARIOS, crossing, head_on, swarm_circle
from .world import World

__all__ = [
    "World",
    "SCENARIOS",
    "crossing",
    "head_on",
    "swarm_circle",
]
