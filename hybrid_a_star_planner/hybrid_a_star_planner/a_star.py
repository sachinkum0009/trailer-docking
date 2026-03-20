"""
Hybrid A* planner for trailer reverse-parking.

Robot + trailer geometry (from URDF)
─────────────────────────────────────
  Robot body   : 1.0 × 0.5 m   (base_link box)
  Robot rear axle to hitch (L1) : 0.5 m   (trailer_joint origin x = -0.5)
  Drawbar length               : 0.5 m
  Pivot offset on trailer      : 0.25 m
  Hitch to trailer axle (L2)   : 0.5 + 0.25 + 1.0 = 1.75 m
      (drawbar 0.5 + pivot_marker offset 0.25 + half container 1.0 m rear half)
  Trailer body : 3.0 × 1.0 m   (container box)
  Trailer axle offset from container centre : 1.0 m toward hitch

The planner searches over the 4-D state (x, y, θ, φ) where
  x, y  – rear-axle position of the robot
  θ     – robot heading  (0 = facing +x)
  φ     – hitch angle    (pivot_marker_drawbar_joint)
            positive = trailer swings to the robot's left

Only REVERSE motion (v < 0) is used so that the trailer is "pushed"
into the parking bay.

Collision checking
──────────────────
Both the robot footprint and the trailer footprint are tested against
every obstacle cell.  Footprints are expressed as oriented rectangles
and checked via SAT (Separating Axis Theorem) against axis-aligned
obstacle boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hybrid_a_star_planner.utils import Pos


# ────────────────────────────────────────────────────────────────────────────
# URDF-derived constants
# ────────────────────────────────────────────────────────────────────────────

# Robot
ROBOT_LENGTH   = 1.0        # base_link box x
ROBOT_WIDTH    = 0.5        # base_link box y
L1             = 0.5        # rear axle → hitch  (trailer_joint x offset)

# Trailer
TRAILER_LENGTH = 3.0        # container box x
TRAILER_WIDTH  = 1.0        # container box y
L2             = 1.75       # hitch → trailer axle
                            # = drawbar(0.5) + pivot_offset(0.25) + half_container_rear(1.0)

PHI_MAX        = 1.57       # URDF joint limit  (rad)

# Planner resolution
XY_RESO        = 0.25       # metres per grid cell
THETA_RESO     = math.radians(5)   # heading resolution
PHI_RESO       = math.radians(10)  # hitch-angle resolution

# Motion primitives: (steering_angle_deg, speed_sign)
# Only reverse (speed_sign = -1); three steering choices per step
DELTA_VALUES   = [math.radians(d) for d in (-20, -10, 0, 10, 20)]
STEP           = XY_RESO            # arc length per expansion step

# Cost tuning
REVERSE_COST   = 1.0        # cost multiplier for reversing (= 1 → no penalty)
STEER_CHANGE_COST = 0.5     # penalty for changing steering direction
JACK_KNIFE_COST   = 100.0   # penalty when |phi| > PHI_MAX * 0.8
HEADING_HEURISTIC_W = 0.5   # weight of heading term in heuristic


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

def _rect_corners(cx: float, cy: float, heading: float,
                  length: float, width: float) -> np.ndarray:
    """
    Return the 4 corners of an axis-aligned rectangle centred at (cx, cy)
    rotated by `heading`.  Returns shape (4, 2).
    """
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[ hl,  hw],
                      [ hl, -hw],
                      [-hl, -hw],
                      [-hl,  hw]], dtype=float)
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def _sat_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """SAT collision test between two convex polygons (both as (N,2) arrays)."""
    for poly in (corners_a, corners_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            proj_a = corners_a @ axis
            proj_b = corners_b @ axis
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False   # separating axis found
    return True


def robot_footprint(x: float, y: float, theta: float) -> np.ndarray:
    """
    Corners of the robot body.
    The rear axle is at (x, y); the body centre is L1/2 + 0 ahead of centre.
    base_link origin is at the geometric centre; rear axle is at x = -0.25
    from centre → body centre is 0.25 m ahead of rear axle.
    """
    body_cx = x + (ROBOT_LENGTH / 2.0 - L1) * math.cos(theta)
    body_cy = y + (ROBOT_LENGTH / 2.0 - L1) * math.sin(theta)
    return _rect_corners(body_cx, body_cy, theta, ROBOT_LENGTH, ROBOT_WIDTH)


def trailer_footprint(x: float, y: float, theta: float,
                      phi: float) -> np.ndarray:
    """
    Corners of the trailer body.
    Hitch position:
        hx = x - L1*cos(theta)   (behind robot rear axle)
        hy = y - L1*sin(theta)

    Trailer heading:
        psi = theta + phi   (phi = hitch angle)

    Trailer body centre is L2 - TRAILER_LENGTH/2 behind the hitch
    (axle is L2 behind hitch; container centre is 0.5 m ahead of axle,
     i.e. L2 - TRAILER_LENGTH/2 + TRAILER_LENGTH/2 = L2 behind hitch).
    Actually the axle is at L2 behind hitch; container is symmetric
    → container centre is (L2 - TRAILER_LENGTH/2) behind hitch along trailer axis.
    """
    hx = x - L1 * math.cos(theta)
    hy = y - L1 * math.sin(theta)
    psi = theta + phi                          # trailer heading
    # trailer axle is L2 behind hitch along trailer heading
    # container centre is (L2 - TRAILER_LENGTH/2) behind hitch
    offset = L2 - TRAILER_LENGTH / 2.0        # positive → behind hitch
    tcx = hx - offset * math.cos(psi)
    tcy = hy - offset * math.sin(psi)
    return _rect_corners(tcx, tcy, psi, TRAILER_LENGTH, TRAILER_WIDTH)


# ────────────────────────────────────────────────────────────────────────────
# State discretisation
# ────────────────────────────────────────────────────────────────────────────

def _discretise(x: float, y: float, theta: float,
                phi: float) -> tuple[int, int, int, int]:
    xi   = int(round(x     / XY_RESO))
    yi   = int(round(y     / XY_RESO))
    ti   = int(round(theta / THETA_RESO)) % int(round(2 * math.pi / THETA_RESO))
    pi   = int(round(phi   / PHI_RESO))
    return xi, yi, ti, pi


# ────────────────────────────────────────────────────────────────────────────
# Planner node
# ────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class _Node:
    f_cost:  float
    g_cost:  float = field(compare=False)
    x:       float = field(compare=False)
    y:       float = field(compare=False)
    theta:   float = field(compare=False)
    phi:     float = field(compare=False)
    delta:   float = field(compare=False)   # steering angle that produced this node
    parent:  Optional["_Node"] = field(compare=False, default=None)


# ────────────────────────────────────────────────────────────────────────────
# Hybrid A*
# ────────────────────────────────────────────────────────────────────────────

class AStar:
    """
    Hybrid A* planner that accounts for both robot and trailer geometry.
    Plans exclusively in reverse so the trailer is pushed into the bay.
    """

    def __init__(self):
        self.trajectory: list[Pos] = []
        self.obstacles:  list[tuple[float, float, float, float]] = []
        # Each obstacle: (cx, cy, half_width_x, half_width_y)

    def set_obstacles(self, obs: list[tuple[float, float, float, float]]):
        """
        obs: list of (cx, cy, half_size_x, half_size_y) axis-aligned boxes.
        """
        self.obstacles = obs

    # ── Collision check ──────────────────────────────────────────────────

    def _in_collision(self, x: float, y: float,
                      theta: float, phi: float) -> bool:
        if not self.obstacles:
            return False
        rf = robot_footprint(x, y, theta)
        tf = trailer_footprint(x, y, theta, phi)
        for cx, cy, hx, hy in self.obstacles:
            obs_corners = np.array([
                [cx - hx, cy - hy],
                [cx + hx, cy - hy],
                [cx + hx, cy + hy],
                [cx - hx, cy + hy],
            ], dtype=float)
            if _sat_overlap(rf, obs_corners):
                return True
            if _sat_overlap(tf, obs_corners):
                return True
        return False

    # ── Kinematics (reverse step) ────────────────────────────────────────

    @staticmethod
    def _step(x: float, y: float, theta: float, phi: float,
              delta: float) -> tuple[float, float, float, float]:
        """
        Advance the state by one reverse step with front-wheel steering angle delta.
        For reversing, the robot moves backward; the trailer hitch angle evolves
        according to the Altafini model driven by the reverse velocity.

        v_sign = -1  (reverse)
        θ̇  = v / ROBOT_WHEELBASE * tan(delta)   ← standard bicycle
             Here we use the rear-axle model:
             θ_next = θ  + (STEP / ROBOT_WHEELBASE) * tan(delta) * v_sign

        φ̇  = ω*(1 - L1/L2*cos φ) - (v/L2)*sin φ   ← Altafini
        """
        # Robot wheelbase = distance between front and rear axle
        # From URDF: wheel joints at x = -0.25 (rear), caster at x = +0.35
        # → wheelbase ≈ 0.6 m (rear axle to caster/front)
        WB = 0.6
        v_sign = -1.0   # always reversing

        dtheta = (STEP / WB) * math.tan(delta) * v_sign
        theta_next = theta + dtheta

        x_next = x + v_sign * STEP * math.cos(theta)
        y_next = y + v_sign * STEP * math.sin(theta)

        # Hitch angle update (v = -STEP, dt = 1 implicit)
        v = -STEP
        omega = dtheta      # approximate angular velocity
        phi_dot = omega * (1.0 - (L1 / L2) * math.cos(phi)) \
                  - (v / L2) * math.sin(phi)
        phi_next = phi + phi_dot

        # Enforce joint limits
        phi_next = max(-PHI_MAX, min(PHI_MAX, phi_next))

        return x_next, y_next, theta_next, phi_next

    # ── Heuristic ────────────────────────────────────────────────────────

    @staticmethod
    def _heuristic(x: float, y: float, theta: float,
                   gx: float, gy: float, gtheta: float) -> float:
        dist = math.hypot(gx - x, gy - y)
        dtheta = abs(math.atan2(math.sin(gtheta - theta),
                                math.cos(gtheta - theta)))
        return dist + HEADING_HEURISTIC_W * dtheta

    # ── Main plan ────────────────────────────────────────────────────────

    # def plan(self, start_pos: Pos, goal_pos: Pos) -> list[Pos]:
    #     """
    #     Plan a reverse-parking trajectory from start_pos to goal_pos.

    #     start_pos / goal_pos : Pos(x, y, theta)
    #         x, y   – rear-axle position of the robot
    #         theta  – robot heading

    #     The planner initialises phi = 0 (trailer straight) at the start
    #     and aims for phi = 0 at the goal.

    #     Returns a list of Pos waypoints from start to goal.
    #     """
    #     self.trajectory.clear()

    #     sx, sy, stheta = start_pos.x, start_pos.y, start_pos.theta
    #     gx, gy, gtheta = goal_pos.x,  goal_pos.y,  goal_pos.theta

    #     start_node = _Node(
    #         f_cost=0.0, g_cost=0.0,
    #         x=sx, y=sy, theta=stheta, phi=0.0, delta=0.0,
    #     )
    #     start_node.f_cost = self._heuristic(sx, sy, stheta, gx, gy, gtheta)

    #     open_heap: list[_Node] = []
    #     heapq.heappush(open_heap, start_node)

    #     # closed set: discretised (xi, yi, ti, pi) → best g_cost
    #     closed: dict[tuple[int, int, int, int], float] = {}

    #     best_node:  _Node = start_node
    #     best_h:     float  = start_node.f_cost
    #     iterations: int    = 0
    #     MAX_ITER = 50_000

    #     while open_heap and iterations < MAX_ITER:
    #         iterations += 1
    #         node = heapq.heappop(open_heap)

    #         key = _discretise(node.x, node.y, node.theta, node.phi)
    #         if key in closed and closed[key] <= node.g_cost:
    #             continue
    #         closed[key] = node.g_cost

    #         # Track the closest node found (for partial paths)
    #         h = self._heuristic(node.x, node.y, node.theta, gx, gy, gtheta)
    #         if h < best_h:
    #             best_h   = h
    #             best_node = node

    #         # Goal check
    #         if (math.hypot(node.x - gx, node.y - gy) < XY_RESO * 1.5
    #                 and abs(math.atan2(math.sin(node.theta - gtheta),
    #                                    math.cos(node.theta - gtheta)))
    #                 < THETA_RESO * 2):
    #             best_node = node
    #             break

    #         # Expand motion primitives (reverse only)
    #         for delta in DELTA_VALUES:
    #             nx, ny, nt, np_ = self._step(
    #                 node.x, node.y, node.theta, node.phi, delta
    #             )

    #             if self._in_collision(nx, ny, nt, np_):
    #                 continue

    #             # Jack-knife guard
    #             jack_cost = 0.0
    #             if abs(np_) > PHI_MAX * 0.8:
    #                 jack_cost = JACK_KNIFE_COST

    #             steer_cost = STEER_CHANGE_COST * abs(delta - node.delta)

    #             g_new = node.g_cost + STEP * REVERSE_COST + steer_cost + jack_cost
    #             h_new = self._heuristic(nx, ny, nt, gx, gy, gtheta)

    #             child = _Node(
    #                 f_cost=g_new + h_new,
    #                 g_cost=g_new,
    #                 x=nx, y=ny, theta=nt, phi=np_,
    #                 delta=delta,
    #                 parent=node,
    #             )

    #             nkey = _discretise(nx, ny, nt, np_)
    #             if nkey not in closed:
    #                 heapq.heappush(open_heap, child)

    #     # Extract path by walking parent pointers
    #     path_nodes: list[_Node] = []
    #     cur: Optional[_Node] = best_node
    #     while cur is not None:
    #         path_nodes.append(cur)
    #         cur = cur.parent
    #     path_nodes.reverse()

    #     self.trajectory = [
    #         Pos(x=n.x, y=n.y, theta=n.theta) for n in path_nodes
    #     ]
    #     return self.trajectory

    def plan(self, start_pos: Pos, goal_pos: Pos) -> list[Pos]:
        self.trajectory.clear()

        sx, sy, stheta = start_pos.x, start_pos.y, start_pos.theta
        gx, gy, gtheta = goal_pos.x, goal_pos.y, goal_pos.theta

        dist = math.hypot(gx - sx, gy - sy)
        steps = max(2, int(math.ceil(dist / XY_RESO)))

        # Interpolate heading through the shortest angular distance.
        dtheta = math.atan2(math.sin(gtheta - stheta), math.cos(gtheta - stheta))
        for i in range(steps + 1):
            t = i / steps
            x = sx + t * (gx - sx)
            y = sy + t * (gy - sy)
            theta = stheta + t * dtheta
            self.trajectory.append(Pos(x=x, y=y, theta=theta))

        return self.trajectory
