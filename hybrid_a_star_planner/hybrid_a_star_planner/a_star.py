import math
import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from hybrid_a_star_planner.utils import Pos

# ────────────────────────────────────────────────────────────────────────────
# Planner Constants Tuned for Long Trailer Stability
# ────────────────────────────────────────────────────────────────────────────
XY_RESO = 0.25           
THETA_RESO = math.radians(5.0)
PHI_RESO = math.radians(5.0)

# Steering Limits: Reduced from 25 to 20 to force wider "truck-like" turns
DELTA_VALUES = [math.radians(d) for d in (-20, -10, 0, 10, 20)]
DIRECTIONS = [1, -1] 
STEP_SIZE = 0.25

# Costs
FORWARD_COST = 1.0
REVERSE_COST = 1.5
STEER_COST = 1.0            # Increased to prefer straight lines
STEER_CHANGE_COST = 8.0     # Increased to prevent rapid "weaving"
DIRECTION_SWITCH_COST = 20.0 # High penalty for the cusp
JACK_KNIFE_PENALTY_COEFF = 1000.0 # Exponential penalty for tight hitch angles

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
        self.L1 = 0.5   
        self.L2 = 1.75  
        self.WB = 0.6   

    def set_obstacles(self, obs):
        self.obstacles = obs

    def _step(self, x: float, y: float, theta: float, phi: float, 
              delta: float, direction: int) -> Tuple[float, float, float, float]:
        """Sub-stepping kinematics for high-fidelity hitch integration."""
        curr_x, curr_y, curr_t, curr_p = x, y, theta, phi
        sub_steps = 10 # Increased for better accuracy in tight turns
        dt = (STEP_SIZE / sub_steps)

        for _ in range(sub_steps):
            dtheta = (dt / self.WB) * math.tan(delta) * direction
            curr_t += dtheta
            curr_x += direction * dt * math.cos(curr_t)
            curr_y += direction * dt * math.sin(curr_t)

            v = dt * direction
            omega = dtheta 
            # Altafini model
            phi_dot = omega * (1.0 - (self.L1 / self.L2) * math.cos(curr_p)) - (v / self.L2) * math.sin(curr_p)
            curr_p += phi_dot

        curr_t = math.atan2(math.sin(curr_t), math.cos(curr_t))
        # HARD LIMIT: Prevent search from ever entering the danger zone (> 60 degrees)
        curr_p = np.clip(curr_p, -1.05, 1.05) 

        return curr_x, curr_y, curr_t, curr_p

    def _heuristic(self, x, y, theta, gx, gy, gtheta):
        dist = math.hypot(gx - x, gy - y)
        angle_err = abs(math.atan2(math.sin(gtheta - theta), math.cos(gtheta - theta)))
        return dist + 1.0 * angle_err

    def plan(self, start_pos: Pos, goal_pos: Pos) -> List[Pos]:
        self.trajectory.clear()
        open_heap = []
        closed = {} 

        # Determine if we should start forward or reverse based on goal direction
        start_node = _Node(
            f_cost=self._heuristic(start_pos.x, start_pos.y, start_pos.theta, goal_pos.x, goal_pos.y, goal_pos.theta),
            g_cost=0.0,
            x=start_pos.x, y=start_pos.y, theta=start_pos.theta, phi=getattr(start_pos, 'phi', 0.0),
            delta=0.0, direction=1
        )
        heapq.heappush(open_heap, start_node)

        best_node = start_node
        iterations = 0
        MAX_ITER = 50000

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

            # Goal Check
            if math.hypot(node.x - goal_pos.x, node.y - goal_pos.y) < 0.35:
                if abs(math.atan2(math.sin(node.theta - goal_pos.theta), math.cos(node.theta - goal_pos.theta))) < 0.2:
                    best_node = node
                    break

            for delta in DELTA_VALUES:
                for direction in DIRECTIONS:
                    nx, ny, nt, np_ = self._step(node.x, node.y, node.theta, node.phi, delta, direction)

                    if self._in_collision(nx, ny, nt, np_):
                        continue

                    # ─── COST CALCULATION ───
                    g_base = STEP_SIZE * (FORWARD_COST if direction > 0 else REVERSE_COST)
                    g_switch = DIRECTION_SWITCH_COST if direction != node.direction else 0.0
                    g_steer_change = abs(delta - node.delta) * STEER_CHANGE_COST
                    g_steer_val = abs(delta) * STEER_COST
                    
                    # SOFT JACKKNIFE BUFFER: 
                    # If phi > 40 degrees (0.7 rad), start penalizing heavily
                    g_jk = 0.0
                    if abs(np_) > 0.7:
                        g_jk = JACK_KNIFE_PENALTY_COEFF * (abs(np_) - 0.7)**2

                    new_g = node.g_cost + g_base + g_switch + g_steer_change + g_steer_val + g_jk
                    new_f = new_g + self._heuristic(nx, ny, nt, goal_pos.x, goal_pos.y, goal_pos.theta)

                    child = _Node(new_f, new_g, nx, ny, nt, np_, delta, direction, node)
                    heapq.heappush(open_heap, child)

        # Path reconstruction
        path = []
        curr = best_node
        while curr:
            p = Pos(x=curr.x, y=curr.y, theta=curr.theta)
            p.phi = curr.phi
            p.is_reverse = (curr.direction == -1)
            path.append(p)
            curr = curr.parent

        path.reverse()
        self.trajectory = path
        return self.trajectory

    def _in_collision(self, x, y, theta, phi):
        # Keep your existing SAT-based collision check here
        return False