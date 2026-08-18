"""End-to-end demo: run and visualise a selected planner and scenario.

Use the preferred-velocity planner as a no-avoidance baseline or the discrete
primitive ORCA planner for collision avoidance.

Usage
-----
    uv run python examples/run_demo.py --planner primitive_orca --scenario crossing --steps 200 --show
    uv run python examples/run_demo.py --scenario swarm_circle --save out.gif
    uv run python examples/run_demo.py --planner reachable_orca --scenario swarm_random --seed 7 --show
    uv run python examples/run_demo.py --planner reachable_orca --scenario swarm_formation --show
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from orca_dubins.planners import (
    PreferredVelocityPlanner,
    PrimitiveOrcaPlanner,
    ReachableOrcaPlanner,
)
from orca_dubins.simulation import (
    FormationGuidance,
    PointGoalGuidance,
    SCENARIOS,
    World,
    assigned_slot_rmse,
    avoidance_intervention_count,
    leader_cross_track_rmse,
    minimum_pairwise_separation,
    swarm_formation_guidance,
)
from orca_dubins.viz import animate, plot_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner",
        choices=("preferred", "primitive_orca", "reachable_orca"),
        default="preferred",
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="crossing")
    parser.add_argument("--seed", type=int, default=0, help="random scenario seed")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--save", type=str, default=None, help="path to .gif/.mp4")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    parser.add_argument("--static", action="store_true", help="static trajectory plot instead of animation")
    args = parser.parse_args()

    if args.scenario in (
        "swarm_random",
        "swarm_formation",
    ):
        agents = SCENARIOS[args.scenario](seed=args.seed)
    else:
        agents = SCENARIOS[args.scenario]()
    guidance = (
        swarm_formation_guidance()
        if args.scenario == "swarm_formation"
        else PointGoalGuidance()
    )
    planners = {
        "preferred": PreferredVelocityPlanner(),
        "primitive_orca": PrimitiveOrcaPlanner(),
        "reachable_orca": ReachableOrcaPlanner(),
    }
    planner = planners[args.planner]
    world = World(
        agents=agents,
        planner=planner,
        guidance=guidance,
        record_diagnostics=True,
        dt=args.dt,
        horizon=args.horizon,
    )
    steps = args.steps if args.steps is not None else (
        600 if args.scenario == "swarm_formation" else 200
    )
    if isinstance(guidance, FormationGuidance):
        world.run_until_complete(steps)
    else:
        world.run(steps)

    print(
        f"minimum separation: "
        f"{minimum_pairwise_separation(world.history):.2f} m"
    )
    print(
        f"avoidance interventions: "
        f"{avoidance_intervention_count(world.history)} agent-frames"
    )
    if isinstance(guidance, FormationGuidance):
        print(
            f"leader cross-track RMSE: "
            f"{leader_cross_track_rmse(world.history, guidance.leader_guidance):.2f} m"
        )
        print(
            f"assigned-slot RMSE: "
            f"{assigned_slot_rmse(world.history, guidance, len(world.history) // 4):.2f} m"
        )
        print(f"mission complete: {world.all_arrived()}")

    radius = {a.id: a.params.radius for a in agents}

    if args.static:
        plot_trajectories(
            world.history,
            agents=agents,
            title=f"{args.scenario} ({planner.name})",
            guidance=guidance,
        )
    else:
        anim = animate(
            world.history,
            agents=agents,
            radius=radius,
            title=f"{args.scenario} ({planner.name})",
            guidance=guidance,
        )
        if args.save:
            anim.save(args.save)
            print(f"saved animation to {args.save}")

    if args.show or (not args.save and not args.static):
        plt.show()
    elif args.static and args.save:
        plt.savefig(args.save)
        print(f"saved figure to {args.save}")


if __name__ == "__main__":
    main()
