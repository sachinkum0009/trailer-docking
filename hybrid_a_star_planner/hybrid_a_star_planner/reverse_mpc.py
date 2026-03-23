"""
MPC for trailer docking
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult
from typing import cast

from hybrid_a_star_planner.utils import Pos


class MPC:
    def __init__(self, dt: float = 0.1, horizon: int = 10):
        self.dt = dt
        self.horizon = horizon  # NOTE: For trailer docking, a longer horizon is needed for better performance

        self.Q = np.diag([1.0, 1.0, 0.5, 3.0])  # State weights [x, y, theta, phi]
        
        self.R = np.diag([0.1, 0.1])  # Control weights [v, omega]
        self.Qf = self.Q * 10  # Final state weight

        self.v_max = 1.0  # max velocity
        self.v_min = -0.5  # min velocity (allowing reverse)
        self.omega_max = 1.0  # max angular velocity


        self.u_prev: NDArray[np.float64] | None = None
        self.last_applied_control: NDArray[np.float64] | None = None
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
        self, position_weight: float, heading_weight: float, control_weight: float, phi_weight: float
    ):
        """Update cost function weights."""
        self.Q = np.diag([position_weight, position_weight, heading_weight, phi_weight])
        self.R = np.diag([control_weight, control_weight])
        self.Qf = self.Q * 10

    def _to_array(self, state: object) -> NDArray[np.float64]:
        if hasattr(state, "x") and hasattr(state, "y") and hasattr(state, "theta"):
            pos_state = cast(Pos, state)
            phi = getattr(pos_state, "phi", 0.0)
            return np.array([pos_state.x, pos_state.y, pos_state.theta, phi], dtype=float)

        arr = np.asarray(state, dtype=float).flatten()
        if arr.size >= 4:
            return arr[:4]
        if arr.size == 3:
            return np.array([arr[0], arr[1], arr[2], 0.0], dtype=float)
        raise ValueError(
            f"State must have at least 3 elements [x, y, theta] (optionally phi), got shape {arr.shape}."
        )

    def _to_pos(self, arr: NDArray[np.float64]) -> Pos:
        return Pos(x=arr[0], y=arr[1], theta=arr[2], phi=arr[3])

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def set_last_applied_control(self, v: float, omega: float):
        self.last_applied_control = np.array([v, omega], dtype=float)

    def robot_dynamics(
        self, state: NDArray[np.float64], control: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Kinematic model for a tractor-trailer system where:
        state = [x, y, theta, phi]
            theta: robot heading
            phi: hitch articulation angle (relative)
        control = [v, omega]
        """
        x, y, theta, phi = state

        # Geometry from the URDF model.
        L1 = 0.5   # rear axle -> hitch
        L2 = 2.75  # hitch -> trailer rear axle (container rear wheel center)

        # Clip controls to physical limits
        v = np.clip(control[0], self.v_min, self.v_max)
        omega = np.clip(control[1], -self.omega_max, self.omega_max)

        # 1. Tractor Dynamics (Standard Unicycle)
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt

        # 2. Hitch articulation dynamics (Altafini-style articulation model)
        phi_dot = omega * (1.0 - (L1 / L2) * np.cos(phi)) - (v / L2) * np.sin(phi)
        phi_next = phi + phi_dot * self.dt

        return np.array([
            x_next,
            y_next,
            self._wrap_angle(theta_next),
            self._wrap_angle(phi_next),
        ])

    def solve_control(self, start_pos, goal_pos) -> NDArray[np.float64]:
        """Solve the MPC optimization problem and return both controls and predicted trajectory."""
        start_arr = self._to_array(start_pos)
        goal_arr = self._to_array(goal_pos)

        # 1. Properly interleaved warm start [v0, w0, v1, w1...]
        if self.u_prev is not None:
            u_shifted = np.vstack([self.u_prev[1:], self.u_prev[-1:]])
            u0 = u_shifted.flatten()
        else:
            dx = goal_arr[0] - start_arr[0]
            dy = goal_arr[1] - start_arr[1]
            goal_heading = np.arctan2(dy, dx)
            heading_error = self._wrap_angle(goal_heading - start_arr[2])
            dist = np.hypot(dx, dy)

            # Seed with a small goal-directed command to avoid zero-control local lock.
            v_seed = float(np.clip(0.6 * dist * np.cos(heading_error), self.v_min, self.v_max))
            if abs(v_seed) < 0.03:
                v_seed = 0.03 if np.cos(heading_error) >= 0.0 else -0.03
            omega_seed = float(np.clip(1.2 * heading_error, -self.omega_max, self.omega_max))
            u0 = np.tile(np.array([v_seed, omega_seed], dtype=float), self.horizon)

        # 2. Properly interleaved bounds
        lower_bounds = np.array([b[0] for b in self.bounds], dtype=float)
        upper_bounds = np.array([b[1] for b in self.bounds], dtype=float)
        u0 = np.clip(u0, lower_bounds, upper_bounds)

        result: OptimizeResult = minimize(
            fun=lambda u: self.cost_function(u, start_arr, goal_arr),
            x0=u0,
            method="SLSQP",  # Keep bounded constrained optimizer for stability.
            bounds=self.bounds,
            options={
                "maxiter": 20,
                "ftol": 5e-3,
                "disp": False,
            },
        )

        if result.x is None or not np.all(np.isfinite(result.x)):
            safe_controls = u0.reshape(self.horizon, 2)
            self.u_prev = safe_controls
            return safe_controls

        # 3. Correct reshape for interleaved data
        optimal_controls = result.x.reshape(self.horizon, 2)

        # SLSQP may hit iteration limits but still return a useful solution.
        if not result.success:
            try:
                if self.cost_function(u0, start_arr, goal_arr) < self.cost_function(result.x, start_arr, goal_arr):
                    optimal_controls = u0.reshape(self.horizon, 2)
            except Exception:
                optimal_controls = u0.reshape(self.horizon, 2)

        self.u_prev = optimal_controls

        return optimal_controls

    def predict_trajectory(
        self, state: NDArray[np.float64], controls_flat: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Predict the trajectory given the initial state and interleaved control sequence."""
        state = self._to_array(state)
        # Ensure extraction matches the interleaved format
        controls = controls_flat.reshape(self.horizon, 2)
        trajectory = np.zeros((self.horizon, 4))
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
        # 1. Setup
        trajectory = self.predict_trajectory(current_state, controls_flat)
        controls = controls_flat.reshape(self.horizon, 2)
        ref = self._to_array(reference)
        
        # Weights for the CHANGE in control [delta_v, delta_omega]
        # A high weight on delta_omega (the second value) specifically kills the "wobble"
        W_delta = np.diag([0.1, 5.0]) 
        v_flip_cost = 3.0
        
        cost = 0.0

        # 2. Loop through the horizon
        for i in range(self.horizon):
            error = trajectory[i] - ref
            # Normalize angles to [-pi, pi]
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
            error[3] = np.arctan2(np.sin(error[3]), np.cos(error[3]))

            # --- Standard Costs ---
            # State error cost (Q) + Control magnitude cost (R)
            cost += error @ self.Q @ error + controls[i] @ self.R @ controls[i]

            # --- NEW: Delta Control (Smoothness) Penalty ---
            if i == 0:
                # Penalize change from the LAST command actually sent to the robot
                if self.last_applied_control is not None:
                    diff = controls[i] - self.last_applied_control
                    cost += diff @ W_delta @ diff
                    if controls[i][0] * self.last_applied_control[0] < 0.0:
                        cost += v_flip_cost
            else:
                # Penalize change between predicted steps in the horizon
                diff = controls[i] - controls[i-1]
                cost += diff @ W_delta @ diff
                if controls[i][0] * controls[i - 1][0] < 0.0:
                    cost += v_flip_cost
            
            # --- NEW: Hitch angle penalty to avoid extreme articulation ---
            phi_abs = abs(trajectory[i, 3])
            if phi_abs > 0.55:  # Start penalizing articulation earlier.
                cost += 1500.0 * (phi_abs - 0.55) ** 2
            if phi_abs > 0.9:  # Strong barrier near dangerous articulation.
                cost += 5000.0 * (phi_abs - 0.9) ** 2

        # 3. Final State Cost
        final_error = trajectory[-1] - ref
        final_error[2] = np.arctan2(np.sin(final_error[2]), np.cos(final_error[2]))
        final_error[3] = np.arctan2(np.sin(final_error[3]), np.cos(final_error[3]))
        cost += final_error @ self.Qf @ final_error

        return cost
