"""
MPC for trailer reverse-parking.

State  : [x, y, theta, phi]
Control: [v, omega]   — v <= 0 during reverse parking

URDF geometry (metres)
  L1 = 0.5   m   robot rear axle → hitch
  L2 = 1.75  m   hitch → trailer axle
  PHI_MAX = 1.57  rad  (URDF joint limit)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from hybrid_a_star_planner.utils import Pos


L1      = 0.5
L2      = 1.75
PHI_MAX = 1.57


class MPC:
    def __init__(self, dt: float = 0.1, horizon: int = 15):
        self.dt      = dt
        self.horizon = horizon

        # State weights [x, y, theta, phi]
        self.Q  = np.diag([1.0, 1.0, 0.5, 3.0])
        self.R  = np.diag([0.1, 0.1])
        self.Qf = self.Q * 10

        # Reverse-only during parking
        self.v_max     = 0.0
        self.v_min     = -0.25
        self.omega_max = 0.5

        self.u_prev    = None
        self.obstacles: list = []
        self._phi: float = 0.0

    # ── Setters ──────────────────────────────────────────────────────────

    def set_physical_constraints(self, v_max: float, v_min: float,
                                 omega_max: float):
        self.v_max     = v_max
        self.v_min     = v_min
        self.omega_max = omega_max

    def set_weights(self, position_weight: float, heading_weight: float,
                    control_weight: float, phi_weight: float = 3.0):
        self.Q  = np.diag([position_weight, position_weight,
                           heading_weight,  phi_weight])
        self.R  = np.diag([control_weight, control_weight])
        self.Qf = self.Q * 10

    def set_obstacles(self, obstacles: list):
        self.obstacles = obstacles

    def update_hitch_angle(self, phi: float):
        self._phi = float(phi)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _to_array(self, state) -> NDArray[np.float64]:
        if hasattr(state, "x") and hasattr(state, "y") and hasattr(state, "theta"):
            return np.array([state.x, state.y, state.theta, 0.0])
        arr = np.asarray(state, dtype=float).flatten()
        if arr.size == 3:
            return np.append(arr, 0.0)
        return arr[:4]

    def _to_pos(self, arr: NDArray[np.float64]) -> Pos:
        return Pos(x=arr[0], y=arr[1], theta=arr[2])

    # ── Kinematic model ──────────────────────────────────────────────────
    #
    #  Unicycle tractor + passive trailer (Altafini 1999)
    #
    #  x_next = x + v·cos(θ)·dt
    #  y_next = y + v·sin(θ)·dt
    #  θ_next = θ + ω·dt
    #  φ_next = φ + [ω·(1 - L1/L2·cos φ) - (v/L2)·sin φ]·dt

    def robot_dynamics(self, state: NDArray[np.float64],
                       control: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y, theta, phi = state
        v     = float(np.clip(control[0], self.v_min, self.v_max))
        omega = float(np.clip(control[1], -self.omega_max, self.omega_max))

        x_next     = x     + v * np.cos(theta) * self.dt
        y_next     = y     + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt

        phi_dot  = omega * (1.0 - (L1 / L2) * np.cos(phi)) \
                   - (v / L2) * np.sin(phi)
        phi_next = float(np.clip(phi + phi_dot * self.dt, -PHI_MAX, PHI_MAX))

        return np.array([x_next, y_next, theta_next, phi_next])

    # ── Trajectory prediction ────────────────────────────────────────────

    def predict_trajectory(self, state: NDArray[np.float64],
                           controls_flat: NDArray[np.float64]) -> NDArray[np.float64]:
        state    = np.asarray(state, dtype=float).flatten()[:4]
        controls = controls_flat.reshape(self.horizon, 2)
        traj     = np.zeros((self.horizon, 4))
        traj[0]  = state
        for i in range(1, self.horizon):
            traj[i] = self.robot_dynamics(traj[i - 1], controls[i - 1])
        return traj

    # ── Collision cost ───────────────────────────────────────────────────

    def compute_collision_cost(self, trajectory: NDArray[np.float64]) -> float:
        if not self.obstacles:
            return 0.0
        cost = 0.0
        robot_radius  = 0.6
        safety_margin = 0.15
        for state in trajectory:
            for ox, oy in self.obstacles:
                dist = np.hypot(state[0] - ox, state[1] - oy)
                if dist < robot_radius + safety_margin:
                    cost += 200.0 * (robot_radius + safety_margin - dist)
        return cost

    # ── Cost function ────────────────────────────────────────────────────

    def cost_function(self, controls_flat: NDArray[np.float64],
                      current_state: NDArray[np.float64],
                      reference: NDArray[np.float64]) -> float:
        traj     = self.predict_trajectory(current_state, controls_flat)
        controls = controls_flat.reshape(self.horizon, 2)
        ref      = np.asarray(reference, dtype=float).flatten()[:4]

        cost = 0.0
        for i in range(self.horizon):
            error    = traj[i] - ref
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
            error[3] = np.arctan2(np.sin(error[3]), np.cos(error[3]))
            cost    += error @ self.Q @ error + controls[i] @ self.R @ controls[i]

            # Jack-knife prevention
            if abs(traj[i, 3]) > PHI_MAX * 0.7:
                cost += 500.0 * (abs(traj[i, 3]) - PHI_MAX * 0.7) ** 2

        final_error    = traj[-1] - ref
        final_error[2] = np.arctan2(np.sin(final_error[2]), np.cos(final_error[2]))
        final_error[3] = np.arctan2(np.sin(final_error[3]), np.cos(final_error[3]))
        cost += final_error @ self.Qf @ final_error
        cost += self.compute_collision_cost(traj)
        return cost

    # ── Solve ────────────────────────────────────────────────────────────

    def solve_control(
        self,
        start_pos,
        goal_pos,
    ) -> tuple[NDArray[np.float64], list[Pos]]:
        start_arr    = self._to_array(start_pos)
        start_arr[3] = self._phi

        goal_arr    = self._to_array(goal_pos)
        goal_arr[3] = 0.0

        if self.u_prev is not None:
            u0 = np.vstack([self.u_prev[1:], self.u_prev[-1:]]).flatten()
        else:
            u0 = np.tile([self.v_min * 0.5, 0.0], self.horizon)

        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.v_min, self.v_max))
            bounds.append((-self.omega_max, self.omega_max))

        result = minimize(
            fun     = lambda u: self.cost_function(u, start_arr, goal_arr),
            x0      = u0,
            method  = "SLSQP",
            bounds  = bounds,
            options = {"maxiter": 30, "ftol": 1e-3},
        )

        optimal_controls = result.x.reshape(self.horizon, 2)
        self.u_prev      = optimal_controls

        predicted_traj = [
            self._to_pos(s)
            for s in self.predict_trajectory(start_arr, result.x)
        ]

        return optimal_controls, predicted_traj