from hybrid_a_star_planner.utils import Pos


class AStar:
    def __init__(self):
        self.trajectory: list[Pos] = []

    def plan(self, start_pos: Pos, goal_pos: Pos) -> list[Pos]:
        """Plans the trajectory using Hybrid A *
        @params
            start_pos: Pos
            goal_pos: Pos
            returns
            trajectory: list
        """
        self.trajectory.clear()
        # use interplate method to generate trajectory with 10 points
        for i in range(10):
            x = start_pos.x + (goal_pos.x - start_pos.x) * i / 9
            y = start_pos.y + (goal_pos.y - start_pos.y) * i / 9
            theta = start_pos.theta + (goal_pos.theta - start_pos.theta) * i / 9
            self.trajectory.append(Pos(x=x, y=y, theta=theta))
        return self.trajectory

    # def plan(self, start_pos: Pos, goal_pos: Pos) -> list[Pos]:
    #     """Plans a U-shaped trajectory using a Quadratic Bezier Curve"""
    #     self.trajectory.clear()
    #     num_points = 10

    #     # 1. Define an offset for the "U" depth
    #     # You can adjust 'dist' to make the U deeper or shallower
    #     dx = goal_pos.x - start_pos.x
    #     dy = goal_pos.y - start_pos.y
    #     dist = 5.0  # Magnitude of the U-turn dip

    #     # 2. Calculate Control Point (Midpoint + Perpendicular Vector)
    #     mid_x = (start_pos.x + goal_pos.x) / 2
    #     mid_y = (start_pos.y + goal_pos.y) / 2
        
    #     # Perpendicular vector (-dy, dx) creates the "sideways" dip
    #     ctrl_x = mid_x - dy * (dist / max(1, (dx**2 + dy**2)**0.5))
    #     ctrl_y = mid_y + dx * (dist / max(1, (dx**2 + dy**2)**0.5))

    #     for i in range(num_points):
    #         t = i / (num_points - 1)
            
    #         # Quadratic Bezier Formula: (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
    #         x = (1 - t)**2 * start_pos.x + 2 * (1 - t) * t * ctrl_x + t**2 * goal_pos.x
    #         y = (1 - t)**2 * start_pos.y + 2 * (1 - t) * t * ctrl_y + t**2 * goal_pos.y
            
    #         # Tangent angle calculation for smooth rotation
    #         # Derivative: 2(1-t)(P1-P0) + 2t(P2-P1)
    #         tx = 2 * (1 - t) * (ctrl_x - start_pos.x) + 2 * t * (goal_pos.x - ctrl_x)
    #         ty = 2 * (1 - t) * (ctrl_y - start_pos.y) + 2 * t * (goal_pos.y - ctrl_y)
    #         import math
    #         theta = math.atan2(ty, tx)

    #         self.trajectory.append(Pos(x=x, y=y, theta=theta))
            
    #     return self.trajectory
