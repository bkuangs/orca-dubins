"""End-to-end demo: run and visualise a selected planner and scenario.

Use the preferred-velocity planner as a no-avoidance baseline or the discrete
primitive ORCA planner for collision avoidance.

Usage
-----
    uv run python examples/run_demo.py --planner primitive_orca --scenario crossing --steps 200 --show
    uv run python examples/run_demo.py --scenario swarm_circle --save out.gif
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from orca_dubins.planners import PreferredVelocityPlanner, PrimitiveOrcaPlanner
from orca_dubins.simulation import SCENARIOS, World
from orca_dubins.viz import animate, plot_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner",
        choices=("preferred", "primitive_orca"),
        default="preferred",
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="crossing")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=float, default=3.0)
    parser.add_argument("--save", type=str, default=None, help="path to .gif/.mp4")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    parser.add_argument("--static", action="store_true", help="static trajectory plot instead of animation")
    args = parser.parse_args()

    agents = SCENARIOS[args.scenario]()
    planners = {
        "preferred": PreferredVelocityPlanner(),
        "primitive_orca": PrimitiveOrcaPlanner(),
    }
    planner = planners[args.planner]
    world = World(agents=agents, planner=planner, dt=args.dt, horizon=args.horizon)
    world.run(args.steps)

    radius = {a.id: a.params.radius for a in agents}

    if args.static:
        plot_trajectories(world.history, agents=agents, title=f"{args.scenario} ({planner.name})")
    else:
        anim = animate(world.history, agents=agents, radius=radius, title=f"{args.scenario} ({planner.name})")
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
