from hybrid_a_star_planner.utils import Pos


class HybridAStar:
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
