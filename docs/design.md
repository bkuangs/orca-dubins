# Design notes: kinodynamically-constrained ORCA + Dubins

This document captures the implemented design of the fixed-wing swarm research
prototype.

## Problem

We want decentralised, fast local collision avoidance for **fixed-wing**
aircraft. ORCA (Optimal Reciprocal Collision Avoidance) is attractive because it
is decentralised and cheap, but it assumes a **holonomic** agent that can pick
any velocity in the plane. Fixed-wing aircraft cannot: they fly at roughly
constant speed and have a bounded turn rate.

### Why naive ORCA → Dubins projection is not enough

- ORCA may choose a velocity that is *theoretically* safe, but by the time the
  aircraft physically turns toward it, the collision can already be unavoidable.
- Projecting ORCA's chosen velocity onto the nearest Dubins-feasible maneuver
  can **invalidate ORCA's safety guarantee** — the projected velocity may fall
  back inside a velocity obstacle.

## The defensible formulation: kinodynamically-constrained ORCA

Constrain ORCA to velocities the aircraft can actually reach over a short
horizon `T_h`, then pick the safest one:

```
v* = argmin_v || v - v_pref ||
subject to
    v ∈ V_ORCA-safe          (collision-avoidance half-planes)
    v ∈ V_Dubins-reachable   (kinodynamic reachable set over T_h)
```

Then **execute the maneuver that actually produces `v*`**.

### Constant-speed fixed-wing model

```
psi_dot = g * tan(phi) / v ,     |phi| <= phi_max
```

So over horizon `T_h` the reachable heading change is bounded. Instead of letting
ORCA consider the entire velocity circle, we restrict it to a **reachable arc**
of headings (constant speed `v`, heading within `± psi_dot_max * T_h`).

This keeps ORCA's biggest advantage — decentralised, fast local avoidance —
without pretending the aircraft is holonomic.

## The practical implementation: discrete control primitives

Rather than continuously optimising over the reachable velocity set:

1. Generate a small set of Dubins/control primitives:
   `max-left, moderate-left, straight, moderate-right, max-right`.
2. Propagate each forward for a short horizon `T_h` under the fixed-wing
   dynamics.
3. Evaluate each rollout against ORCA's collision constraints (`V_ORCA-safe`).
4. Select the **feasible** maneuver whose resulting velocity is closest to the
   preferred mission velocity `v_pref`.

Because every primitive is Dubins-feasible by construction, the
`V_Dubins-reachable` constraint is automatically satisfied; only ORCA safety
needs to be checked.

## How this maps to the code

| Concept | Module | Status |
|---|---|---|
| Aircraft types / states | `orca_dubins.types` | done |
| Constant-speed kinematics, turn-rate limit | `orca_dubins.dynamics` | done (substrate) |
| Velocity obstacles | `orca_dubins.orca.velocity_obstacle` | done |
| ORCA half-planes / safe set | `orca_dubins.orca.orca` | done |
| Dubins reachable set | `orca_dubins.dubins.primitives.dubins_reachable_velocities` | done |
| Control primitives | `orca_dubins.dubins.primitives` | done |
| Classic Dubins shortest path | `orca_dubins.dubins.path` | done |
| Planner interface | `orca_dubins.planners.base` | done |
| No-avoidance baseline | `orca_dubins.planners.baseline` | done (runnable) |
| Kinodynamic ORCA planner | `orca_dubins.planners.reachable_orca` | done |
| Control-primitive planner | `orca_dubins.planners.primitive_orca` | done |
| Mission and formation guidance | `orca_dubins.simulation.guidance` | done |
| Simulation metrics | `orca_dubins.simulation.metrics` | done |
| Simulation loop / scenarios | `orca_dubins.simulation` | done |
| Matplotlib viz | `orca_dubins.viz` | done |

## Open questions to revisit while implementing

- **Neighbour filtering:** sensing range / max number of neighbours per agent.
- **Fallback when no primitive is ORCA-feasible:** hardest-braking turn? hold?
  min-penetration velocity?
- **Responsibility split:** reciprocal (0.5) vs. treating some agents as
  non-cooperative (1.0).
- **Speed variation:** the current model is strictly constant speed; do we ever
  want a small speed envelope?
- **3D extension:** altitude/climb-rate as a later axis (types are structured to
  allow it).
