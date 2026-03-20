"""
MPC for trailer docking
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult

from hybrid_a_star_planner.utils import Pos


class MPC:
    def __init__(self, dt: float = 0.1, horizon: int = 10):
        self.dt = dt
        self.horizon = horizon  # NOTE: For trailer docking, a longer horizon is needed for better performance

        self.Q = np.diag([1.0, 1.0, 0.5])  # State weights [x, y, theta]
        self.R = np.diag([0.1, 0.1])  # Control weights [v, omega]
        self.Qf = self.Q * 10  # Final state weight

        self.v_max = 1.0  # max velocity
        self.v_min = -0.5  # min velocity (allowing reverse)
        self.omega_max = 1.0  # max angular velocity

        self.u_prev = None
        self.obstacles = []

        self.bounds = []
        for _ in range(self.horizon):
            self.bounds.append((self.v_min, self.v_max))
            self.bounds.append((-self.omega_max, self.omega_max))

    def set_physical_constraints(self, v_max: float, v_min: float, omega_max: float):
        """Update physical constraints and corresponding optimization bounds."""
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max

        self.bounds.clear()
        for _ in range(self.horizon):
            self.bounds.append((self.v_min, self.v_max))
            self.bounds.append((-self.omega_max, self.omega_max))

    def set_weights(
        self, position_weight: float, heading_weight: float, control_weight: float
    ):
        """Update cost function weights."""
        self.Q = np.diag([position_weight, position_weight, heading_weight])
        self.R = np.diag([control_weight, control_weight])
        self.Qf = self.Q * 10

    def _to_array(self, state) -> NDArray[np.float64]:
        if hasattr(state, "x") and hasattr(state, "y") and hasattr(state, "theta"):
            return np.array([state.x, state.y, state.theta])
        arr = np.asarray(state)
        if arr.ndim == 0:
            return arr
        return arr.flatten()[:3]

    def _to_pos(self, arr: NDArray[np.float64]) -> Pos:
        return Pos(x=arr[0], y=arr[1], theta=arr[2])

    def robot_dynamics(
        self, state: NDArray[np.float64], control: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Simple unicycle model dynamics for the robot.
        """
        x, y, theta = state
        v, omega = (
            np.clip(control[0], self.v_min, self.v_max),
            np.clip(control[1], -self.omega_max, self.omega_max),
        )
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt
        return np.array([x_next, y_next, theta_next])

    def solve_control(self, start_pos, goal_pos) -> NDArray[np.float64]:
        """Solve the MPC optimization problem and return both controls and predicted trajectory."""
        start_arr = self._to_array(start_pos)
        goal_arr = self._to_array(goal_pos)

        # 1. Properly interleaved warm start [v0, w0, v1, w1...]
        if self.u_prev is not None:
            u_shifted = np.vstack([self.u_prev[1:], self.u_prev[-1:]])
            u0 = u_shifted.flatten()
        else:
            u0 = np.zeros(self.horizon * 2)

        # 2. Properly interleaved bounds
        result: OptimizeResult = minimize(
            fun=lambda u: self.cost_function(u, start_arr, goal_arr),
            x0=u0,
            method="SLSQP",  # SLSQP handles bounds and is generally faster for MPC problems than COBYLA
            bounds=self.bounds,
            options={
                "maxiter": 20,
                "ftol": 1e-3,
                "disp": False,
            },  # Reduced iterations for real-time speed
        )

        # 3. Correct reshape for interleaved data
        optimal_controls = result.x.reshape(self.horizon, 2)
        self.u_prev = optimal_controls

        return optimal_controls

    def predict_trajectory(
        self, state: NDArray[np.float64], controls_flat: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Predict the trajectory given the initial state and interleaved control sequence."""
        state = np.asarray(state).flatten()[:3]
        # Ensure extraction matches the interleaved format
        controls = controls_flat.reshape(self.horizon, 2)
        trajectory = np.zeros((self.horizon, 3))
        trajectory[0] = state
        for i in range(1, self.horizon):
            trajectory[i] = self.robot_dynamics(trajectory[i - 1], controls[i - 1])
        return trajectory

    def cost_function(
        self,
        controls_flat: NDArray[np.float64],
        current_state: NDArray[np.float64],
        reference: NDArray[np.float64],
    ) -> float:
        # Ensure extraction matches the interleaved format
        trajectory = self.predict_trajectory(current_state, controls_flat)
        controls = controls_flat.reshape(self.horizon, 2)

        ref = np.asarray(reference).flatten()[:3]
        cost = 0.0

        for i in range(self.horizon):
            error = trajectory[i] - ref
            # Normalize angle to [-pi, pi]
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))

            # State cost + Control effort cost (state error + control error)
            cost += error @ self.Q @ error + controls[i] @ self.R @ controls[i]

        final_error = trajectory[-1] - ref
        final_error[2] = np.arctan2(np.sin(final_error[2]), np.cos(final_error[2]))
        cost += final_error @ self.Qf @ final_error

        return cost
