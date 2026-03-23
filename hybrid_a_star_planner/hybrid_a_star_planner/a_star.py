import math
import heapq
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from hybrid_a_star_planner.utils import Pos

# --- Updated Constants for Bi-directional Planning ---
DIRECTION_SWITCH_COST = 5.0  # High cost to encourage only one direction change
REVERSE_COST = 1.5           # Penalty for reversing to prefer forward when possible
FORWARD_COST = 1.0
STEER_CHANGE_COST = 0.2
JACK_KNIFE_COST = 150.0

# Motion primitives: (steering_angle, direction_sign)
# direction_sign: 1 for forward, -1 for reverse
DELTA_VALUES = [math.radians(d) for d in (-20, -10, 0, 10, 20)]
DIRECTIONS = [1, -1] 

@dataclass(order=True)
class _Node:
    f_cost:  float
    g_cost:  float = field(compare=False)
    x:       float = field(compare=False)
    y:       float = field(compare=False)
    theta:   float = field(compare=False)
    phi:     float = field(compare=False)
    delta:   float = field(compare=False)
    direction: int = field(compare=False) # 1 or -1
    parent:  Optional["_Node"] = field(compare=False, default=None)

class AStar:
    def __init__(self):
        self.trajectory: list[Pos] = []
        self.obstacles: list[tuple[float, float, float, float]] = []

    def set_obstacles(self, obs):
        self.obstacles = obs

    @staticmethod
    def _step(x, y, theta, phi, delta, v_sign):
        """Kinematics supporting both forward and reverse."""
        WB = 0.6
        STEP = 0.25 # Matches XY_RESO
        
        # Robot Heading
        dtheta = (STEP / WB) * math.tan(delta) * v_sign
        theta_next = theta + dtheta

        # Robot Position
        x_next = x + v_sign * STEP * math.cos(theta)
        y_next = y + v_sign * STEP * math.sin(theta)

        # Altafini Trailer Dynamics
        # Linear velocity v = STEP * v_sign
        v = STEP * v_sign
        omega = dtheta 
        L1, L2 = 0.5, 1.75
        
        phi_dot = omega * (1.0 - (L1 / L2) * math.cos(phi)) - (v / L2) * math.sin(phi)
        phi_next = phi + phi_dot
        
        # Clamp phi to joint limits
        PHI_MAX = 1.57
        phi_next = max(-PHI_MAX, min(PHI_MAX, phi_next))

        return x_next, y_next, theta_next, phi_next

    def _heuristic(self, x, y, theta, gx, gy, gtheta):
        dist = math.hypot(gx - x, gy - y)
        angle_err = abs(math.atan2(math.sin(gtheta - theta), math.cos(gtheta - theta)))
        return dist + 0.5 * angle_err

    def plan(self, start_pos: Pos, goal_pos: Pos) -> list[Pos]:
        self.trajectory.clear()
        
        open_heap = []
        closed = {} # (xi, yi, ti, pi) -> g_cost
        
        start_node = _Node(
            f_cost=self._heuristic(start_pos.x, start_pos.y, start_pos.theta, goal_pos.x, goal_pos.y, goal_pos.theta),
            g_cost=0.0,
            x=start_pos.x, y=start_pos.y, theta=start_pos.theta, phi=0.0,
            delta=0.0, direction=1
        )
        heapq.heappush(open_heap, start_node)

        best_node = start_node
        iterations = 0
        MAX_ITER = 30000

        while open_heap and iterations < MAX_ITER:
            iterations += 1
            node = heapq.heappop(open_heap)

            # Discretize for closed set
            key = (int(round(node.x/0.25)), int(round(node.y/0.25)), 
                   int(round(node.theta/math.radians(5))) % 72,
                   int(round(node.phi/math.radians(10))))
            
            if key in closed and closed[key] <= node.g_cost:
                continue
            closed[key] = node.g_cost

            # Goal Check (Tighter for parking)
            if math.hypot(node.x - goal_pos.x, node.y - goal_pos.y) < 0.3:
                if abs(math.atan2(math.sin(node.theta - goal_pos.theta), math.cos(node.theta - goal_pos.theta))) < 0.2:
                    best_node = node
                    break

            # Expand Forward and Reverse Primitives
            for delta in DELTA_VALUES:
                for v_sign in DIRECTIONS:
                    nx, ny, nt, np_ = self._step(node.x, node.y, node.theta, node.phi, delta, v_sign)
                    
                    if self._in_collision(nx, ny, nt, np_):
                        continue

                    # --- Cost Calculation ---
                    step_cost = 0.25 * (FORWARD_COST if v_sign > 0 else REVERSE_COST)
                    
                    # Penalty for changing direction (creates the 'cusp' in your drawing)
                    switch_cost = DIRECTION_SWITCH_COST if v_sign != node.direction else 0.0
                    
                    # Penalty for steering change
                    steer_cost = STEER_CHANGE_COST * abs(delta - node.delta)
                    
                    # Penalty for approaching jackknife
                    jk_cost = JACK_KNIFE_COST if abs(np_) > 1.2 else 0.0

                    new_g = node.g_cost + step_cost + switch_cost + steer_cost + jk_cost
                    new_f = new_g + self._heuristic(nx, ny, nt, goal_pos.x, goal_pos.y, goal_pos.theta)

                    child = _Node(new_f, new_g, nx, ny, nt, np_, delta, v_sign, node)
                    heapq.heappush(open_heap, child)

        # Reconstruct Path
        path = []
        curr = best_node
        while curr:
            path.append(Pos(
                x=curr.x, y=curr.y, theta=curr.theta, 
                phi=curr.phi, 
                is_reverse=(curr.direction == -1)
            ))
            curr = curr.parent
        
        path.reverse()
        self.trajectory = path
        return self.trajectory

    def _in_collision(self, x, y, theta, phi):
        # Implementation of your SAT check from previous code
        # (Omitted here for brevity, keep your existing SatOverlap logic)
        return False