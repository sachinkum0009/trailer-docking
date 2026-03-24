import math
import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from hybrid_a_star_planner.utils import Pos

# ────────────────────────────────────────────────────────────────────────────
# Planner Constants Tuned for Curvy S-Curves
# ────────────────────────────────────────────────────────────────────────────
XY_RESO = 0.20           # Finer resolution for smoother curves
THETA_RESO = math.radians(3.0)
PHI_RESO = math.radians(3.0)

# More steering options = smoother curves
DELTA_VALUES = [math.radians(d) for d in (-25, -15, -7, 0, 7, 15, 25)]
DIRECTIONS = [1, -1] 
STEP_SIZE = 0.20

# Costs tuned for "Curviness"
FORWARD_COST = 1.0
REVERSE_COST = 1.2       # Closer to forward cost to allow easier reversing
STEER_COST = 0.1         # VERY low penalty for steering to encourage curving
STEER_CHANGE_COST = 2.0  # Reduced to allow fluid steering transitions
DIRECTION_SWITCH_COST = 15.0 
CURVE_COST = 0.5         # Penalty for 0 steering to "force" curves if they help
JACK_KNIFE_PENALTY_COEFF = 1500.0 

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
        self.L2 = 2.75  
        self.WB = 0.6   

    def set_obstacles(self, obs):
        self.obstacles = obs

    def _step(self, x: float, y: float, theta: float, phi: float, 
              delta: float, direction: int) -> Tuple[float, float, float, float]:
        curr_x, curr_y, curr_t, curr_p = x, y, theta, phi
        sub_steps = 10 
        dt = (STEP_SIZE / sub_steps)

        for _ in range(sub_steps):
            dtheta = (dt / self.WB) * math.tan(delta) * direction
            curr_t += dtheta
            curr_x += direction * dt * math.cos(curr_t)
            curr_y += direction * dt * math.sin(curr_t)

            v = dt * direction
            omega = dtheta 
            phi_dot = omega * (1.0 - (self.L1 / self.L2) * math.cos(curr_p)) - (v / self.L2) * math.sin(curr_p)
            curr_p += phi_dot

        curr_t = math.atan2(math.sin(curr_t), math.cos(curr_t))
        curr_p = np.clip(curr_p, -1.05, 1.05) 

        return curr_x, curr_y, curr_t, curr_p

    def _heuristic(self, x, y, theta, gx, gy, gtheta):
        dist = math.hypot(gx - x, gy - y)
        # Weight angle more heavily to force the robot to "aim" correctly early on
        angle_err = abs(math.atan2(math.sin(gtheta - theta), math.cos(gtheta - theta)))
        return dist + 1.5 * angle_err

    def plan(self, start_pos: Pos, goal_pos: Pos) -> List[Pos]:
        self.trajectory.clear()
        open_heap = []
        closed = {} 

        start_node = _Node(
            f_cost=self._heuristic(start_pos.x, start_pos.y, start_pos.theta, goal_pos.x, goal_pos.y, goal_pos.theta),
            g_cost=0.0,
            x=start_pos.x, y=start_pos.y, theta=start_pos.theta, phi=getattr(start_pos, 'phi', 0.0),
            delta=0.0, direction=1
        )
        heapq.heappush(open_heap, start_node)

        best_node = start_node
        iterations = 0
        MAX_ITER = 60000 # Increased as smoother curves require more exploration

        while open_heap and iterations < MAX_ITER:
            iterations += 1
            node = heapq.heappop(open_heap)

            key = (int(round(node.x / XY_RESO)), 
                   int(round(node.y / XY_RESO)), 
                   int(round(node.theta / THETA_RESO)) % 120,
                   int(round(node.phi / PHI_RESO)))

            if key in closed and closed[key] <= node.g_cost:
                continue
            closed[key] = node.g_cost

            if math.hypot(node.x - goal_pos.x, node.y - goal_pos.y) < 0.30:
                if abs(math.atan2(math.sin(node.theta - goal_pos.theta), math.cos(node.theta - goal_pos.theta))) < 0.15:
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
                    
                    # Curved preference: penalize delta=0 slightly to encourage "S" shapes
                    g_curve = CURVE_COST if abs(delta) < 0.01 else 0.0
                    
                    g_jk = 0.0
                    if abs(np_) > 0.65: # Even tighter buffer for "curvy" safety
                        g_jk = JACK_KNIFE_PENALTY_COEFF * (abs(np_) - 0.65)**2

                    new_g = node.g_cost + g_base + g_switch + g_steer_change + g_curve + g_jk
                    new_f = new_g + self._heuristic(nx, ny, nt, goal_pos.x, goal_pos.y, goal_pos.theta)

                    child = _Node(new_f, new_g, nx, ny, nt, np_, delta, direction, node)
                    heapq.heappush(open_heap, child)

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
        return False