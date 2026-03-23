import math
import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from hybrid_a_star_planner.utils import Pos

# ────────────────────────────────────────────────────────────────────────────
# Planner Constants for S-Curve Smoothness
# ────────────────────────────────────────────────────────────────────────────
XY_RESO = 0.25           # Meters
THETA_RESO = math.radians(5.0)
PHI_RESO = math.radians(5.0)

# Costs: High penalties for "wiggling" or "switching"
FORWARD_COST = 1.0
REVERSE_COST = 1.5       # Slightly prefer forward
STEER_COST = 0.5         # Penalty for turning (prefers straight)
STEER_CHANGE_COST = 5.0  # Penalty for twitching wheels (Smoothness)
DIRECTION_SWITCH_COST = 15.0 # High penalty to force only ONE cusp (S-curve)
JACK_KNIFE_COST = 200.0

# Motion Primitives
DELTA_VALUES = [math.radians(d) for d in (-25, -12, 0, 12, 25)]
DIRECTIONS = [1, -1] # 1: Forward, -1: Reverse
STEP_SIZE = 0.25

@dataclass(order=True)
class _Node:
    f_cost:  float
    g_cost:  float = field(compare=False)
    x:       float = field(compare=False)
    y:       float = field(compare=False)
    theta:   float = field(compare=False)
    phi:     float = field(compare=False)
    delta:   float = field(compare=False)
    direction: int = field(compare=False)
    parent:  Optional["_Node"] = field(compare=False, default=None)

class AStar:
    def __init__(self):
        self.trajectory: List[Pos] = []
        self.obstacles: List[Tuple[float, float, float, float]] = []
        self.L1 = 0.5   # Robot axle to hitch
        self.L2 = 1.75  # Hitch to trailer axle
        self.WB = 0.6   # Robot wheelbase

    def set_obstacles(self, obs):
        self.obstacles = obs

    def _step(self, x: float, y: float, theta: float, phi: float, 
              delta: float, direction: int) -> Tuple[float, float, float, float]:
        """
        Sub-stepping kinematics for smooth integration of the trailer angle.
        """
        curr_x, curr_y, curr_t, curr_p = x, y, theta, phi
        sub_steps = 5
        dt = (STEP_SIZE / sub_steps)

        for _ in range(sub_steps):
            # Robot Bicycle Model
            dtheta = (dt / self.WB) * math.tan(delta) * direction
            curr_t += dtheta
            curr_x += direction * dt * math.cos(curr_t)
            curr_y += direction * dt * math.sin(curr_t)

            # Trailer Altafini Model
            v = dt * direction
            omega = dtheta 
            phi_dot = omega * (1.0 - (self.L1 / self.L2) * math.cos(curr_p)) - (v / self.L2) * math.sin(curr_p)
            curr_p += phi_dot

        # Normalize angles
        curr_t = math.atan2(math.sin(curr_t), math.cos(curr_t))
        curr_p = np.clip(curr_p, -1.57, 1.57)

        return curr_x, curr_y, curr_t, curr_p

    def _heuristic(self, x, y, theta, gx, gy, gtheta):
        dist = math.hypot(gx - x, gy - y)
        # Heading alignment is critical for the "Parking" stage
        angle_err = abs(math.atan2(math.sin(gtheta - theta), math.cos(gtheta - theta)))
        return dist + 0.8 * angle_err

    def plan(self, start_pos: Pos, goal_pos: Pos) -> List[Pos]:
        self.trajectory.clear()
        
        open_heap = []
        # Key: (xi, yi, ti, pi)
        closed = {} 

        # Initial node assumes robot is moving forward initially
        start_node = _Node(
            f_cost=self._heuristic(start_pos.x, start_pos.y, start_pos.theta, goal_pos.x, goal_pos.y, goal_pos.theta),
            g_cost=0.0,
            x=start_pos.x, y=start_pos.y, theta=start_pos.theta, phi=getattr(start_pos, 'phi', 0.0),
            delta=0.0, direction=1
        )
        heapq.heappush(open_heap, start_node)

        best_node = start_node
        iterations = 0
        MAX_ITER = 40000

        while open_heap and iterations < MAX_ITER:
            iterations += 1
            node = heapq.heappop(open_heap)

            key = (int(round(node.x / XY_RESO)), 
                   int(round(node.y / XY_RESO)), 
                   int(round(node.theta / THETA_RESO)) % 72,
                   int(round(node.phi / PHI_RESO)))

            if key in closed and closed[key] <= node.g_cost:
                continue
            closed[key] = node.g_cost

            # Goal Check: within radius and aligned heading
            if math.hypot(node.x - goal_pos.x, node.y - goal_pos.y) < 0.4:
                heading_error = abs(math.atan2(math.sin(node.theta - goal_pos.theta), math.cos(node.theta - goal_pos.theta)))
                if heading_error < 0.2:
                    best_node = node
                    break

            # Branching: Try all steering angles for BOTH Forward and Reverse
            for delta in DELTA_VALUES:
                for direction in DIRECTIONS:
                    nx, ny, nt, np_ = self._step(node.x, node.y, node.theta, node.phi, delta, direction)

                    if self._in_collision(nx, ny, nt, np_):
                        continue

                    # --- Cost Function ───
                    # 1. Distance cost
                    g_step = STEP_SIZE * (FORWARD_COST if direction > 0 else REVERSE_COST)
                    
                    # 2. Smoothness cost (Penalize changing steer angle)
                    g_steer = abs(delta - node.delta) * STEER_CHANGE_COST
                    
                    # 3. Direction change cost (The Cusp penalty)
                    g_switch = DIRECTION_SWITCH_COST if direction != node.direction else 0.0
                    
                    # 4. Jackknife penalty
                    g_jk = JACK_KNIFE_COST if abs(np_) > 1.2 else 0.0

                    new_g = node.g_cost + g_step + g_steer + g_switch + g_jk
                    new_f = new_g + self._heuristic(nx, ny, nt, goal_pos.x, goal_pos.y, goal_pos.theta)

                    child = _Node(new_f, new_g, nx, ny, nt, np_, delta, direction, node)
                    heapq.heappush(open_heap, child)

        # Reconstruct path from back to front
        path = []
        curr = best_node
        while curr:
            # We save the direction metadata so the MPC knows when to shift gears
            p = Pos(x=curr.x, y=curr.y, theta=curr.theta)
            p.phi = curr.phi
            p.is_reverse = (curr.direction == -1)
            path.append(p)
            curr = curr.parent

        path.reverse()
        self.trajectory = path
        return self.trajectory

    def _in_collision(self, x, y, theta, phi):
        """Placeholder for your SAT-based footprint check."""
        # Use your existing robot_footprint and trailer_footprint logic here.
        return False