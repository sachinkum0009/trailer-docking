"""
MPC for trailer docking
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from hybrid_a_star_planner.utils import Pos


class MPC:
    def __init__(self, dt: float = 0.1, horizon: int = 10):
        self.dt = dt
        self.horizon = horizon

        self.Q = np.diag([1.0, 1.0, 0.5])
        self.R = np.diag([0.1, 0.1])
        self.Qf = self.Q * 10

        self.v_max = 1.0
        self.v_min = -0.5
        self.omega_max = 1.0

        self.u_prev = None
        self.obstacles = []

    def set_physical_constraints(self, v_max: float, v_min: float, omega_max: float):
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max

    def set_weights(
        self, position_weight: float, heading_weight: float, control_weight: float
    ):
        self.Q = np.diag([position_weight, position_weight, heading_weight])
        self.R = np.diag([control_weight, control_weight])
        self.Qf = self.Q * 10

    def set_obstacles(self, obstacles: list):
        self.obstacles = obstacles

    def _to_array(self, state) -> NDArray[np.float64]:
        if hasattr(state, "x") and hasattr(state, "y") and hasattr(state, "theta"):
            return np.array([state.x, state.y, state.theta])
        arr = np.asarray(state)
        if arr.ndim == 0:
            return arr
        return arr.flatten()[:3]

    def _to_pos(self, arr: NDArray[np.float64]) -> Pos:
        return Pos(x=arr[0], y=arr[1], theta=arr[2])

    # def predict_trajectory(self, state, controls) -> NDArray[np.float64]:
    #     state = np.asarray(state).flatten()[:3]
    #     controls = np.array(controls).reshape(self.horizon, 2)
    #     trajectory = np.zeros((self.horizon, 3))
    #     trajectory[0] = state
    #     for i in range(1, self.horizon):
    #         trajectory[i] = self.robot_dynamics(trajectory[i - 1], controls[i - 1])
    #     return trajectory

    def robot_dynamics(
        self, state: NDArray[np.float64], control: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        x, y, theta = state
        v, omega = (
            np.clip(control[0], self.v_min, self.v_max),
            np.clip(control[1], -self.omega_max, self.omega_max),
        )
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt
        return np.array([x_next, y_next, theta_next])

    def compute_collision_cost(self, trajectory: NDArray[np.float64]) -> float:
        if not self.obstacles:
            return 0.0
        cost = 0.0
        robot_radius = 0.25
        for state in trajectory:
            for ox, oy in self.obstacles:
                dist = np.sqrt((state[0] - ox) ** 2 + (state[1] - oy) ** 2)
                if dist < robot_radius + 0.1:
                    cost += 100.0 * (robot_radius + 0.1 - dist)
        return cost

    # def cost_function(self, controls, current_state, reference):
    #     trajectory = self.predict_trajectory(current_state, controls)
    #     controls = np.array(controls).reshape(self.horizon, 2)

    #     ref = np.asarray(reference).flatten()[:3]
    #     ref = np.tile(ref, (self.horizon, 1))

    #     cost = 0.0
    #     for i in range(self.horizon):
    #         error = trajectory[i] - ref[i]
    #         error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    #         cost += error @ self.Q @ error + controls[i] @ self.R @ controls[i]

    #     final_error = trajectory[-1] - ref[-1]
    #     final_error[2] = np.arctan2(np.sin(final_error[2]), np.cos(final_error[2]))
    #     cost += final_error @ self.Qf @ final_error
    #     cost += self.compute_collision_cost(trajectory)

    #     return cost

    def solve(self, start_pos, goal_pos):
        start_arr = self._to_array(start_pos)
        goal_arr = self._to_array(goal_pos)

        if self.u_prev is not None:
            u0 = np.vstack([self.u_prev[1:], self.u_prev[-1:]]).flatten()
        else:
            u0 = np.zeros(self.horizon * 2)

        bounds = [(self.v_min, self.v_max)] * self.horizon + [
            (-self.omega_max, self.omega_max)
        ] * self.horizon

        result = minimize(
            fun=lambda u: self.cost_function(u, start_arr, goal_arr),
            x0=u0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-3},
        )

        optimal_controls = result.x.reshape(self.horizon, 2)
        self.u_prev = optimal_controls
        predicted_traj = self.predict_trajectory(start_arr, result.x)

        return [self._to_pos(state) for state in predicted_traj]
    
   
    # def solve_control(self, start_pos: Pos, goal_pos: Pos) -> tuple[NDArray[np.float64], list[Pos]]:
    #     """
    #     Solves the MPC optimization problem.
    #     Returns:
    #         optimal_controls: NDArray of shape (horizon, 2) -> [v, omega]
    #         predicted_traj: List of Pos objects for visualization
    #     """
    #     start_arr = self._to_array(start_pos)
    #     goal_arr = self._to_array(goal_pos)

    #     # Warm start: use the previous solution shifted by one step
    #     if self.u_prev is not None:
    #         u0 = np.vstack([self.u_prev[1:], self.u_prev[-1:]]).flatten()
    #     else:
    #         u0 = np.zeros(self.horizon * 2)

    #     # Bounds for [v_0, v_1... v_N, omega_0, omega_1... omega_N]
    #     bounds = [(self.v_min, self.v_max)] * self.horizon + [
    #         (-self.omega_max, self.omega_max)
    #     ] * self.horizon

    #     result = minimize(
    #         fun=lambda u: self.cost_function(u, start_arr, goal_arr),
    #         x0=u0,
    #         method="SLSQP",
    #         bounds=bounds,
    #         options={"maxiter": 50, "ftol": 1e-3},
    #     )

    #     # Reshape the flat result back into (horizon, 2)
    #     optimal_controls = result.x.reshape(2, self.horizon).T
    #     self.u_prev = optimal_controls
        
    #     # Generate the trajectory that these controls will actually produce
    #     predicted_traj_arr = self.predict_trajectory(start_arr, result.x)
    #     predicted_traj = [self._to_pos(state) for state in predicted_traj_arr]

    #     return optimal_controls, predicted_traj

    def solve_control(self, start_pos, goal_pos):
        start_arr = self._to_array(start_pos)
        goal_arr = self._to_array(goal_pos)

        # 1. Properly interleaved warm start [v0, w0, v1, w1...]
        if self.u_prev is not None:
            u_shifted = np.vstack([self.u_prev[1:], self.u_prev[-1:]])
            u0 = u_shifted.flatten()
        else:
            u0 = np.zeros(self.horizon * 2)

        # 2. Properly interleaved bounds
        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.v_min, self.v_max))
            bounds.append((-self.omega_max, self.omega_max))

        result = minimize(
            fun=lambda u: self.cost_function(u, start_arr, goal_arr),
            x0=u0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 20, "ftol": 1e-3}, # Reduced iterations for real-time speed
        )

        # 3. Correct reshape for interleaved data
        optimal_controls = result.x.reshape(self.horizon, 2)
        self.u_prev = optimal_controls
        
        predicted_traj_arr = self.predict_trajectory(start_arr, result.x)
        predicted_traj = [self._to_pos(state) for state in predicted_traj_arr]

        return optimal_controls, predicted_traj

    def predict_trajectory(self, state, controls_flat) -> NDArray[np.float64]:
        state = np.asarray(state).flatten()[:3]
        # Ensure extraction matches the interleaved format
        controls = controls_flat.reshape(self.horizon, 2)
        trajectory = np.zeros((self.horizon, 3))
        trajectory[0] = state
        for i in range(1, self.horizon):
            trajectory[i] = self.robot_dynamics(trajectory[i - 1], controls[i - 1])
        return trajectory

    def cost_function(self, controls_flat, current_state, reference):
        # Ensure extraction matches the interleaved format
        trajectory = self.predict_trajectory(current_state, controls_flat)
        controls = controls_flat.reshape(self.horizon, 2)

        ref = np.asarray(reference).flatten()[:3]
        cost = 0.0
        
        for i in range(self.horizon):
            error = trajectory[i] - ref
            # Normalize angle to [-pi, pi]
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
            
            # State cost + Control effort cost
            cost += error @ self.Q @ error + controls[i] @ self.R @ controls[i]

        final_error = trajectory[-1] - ref
        final_error[2] = np.arctan2(np.sin(final_error[2]), np.cos(final_error[2]))
        cost += final_error @ self.Qf @ final_error
        cost += self.compute_collision_cost(trajectory)

        return cost
