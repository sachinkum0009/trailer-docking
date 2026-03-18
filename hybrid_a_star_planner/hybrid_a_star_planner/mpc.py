"""
MPC for trailer docking/parking.

State vector: [x, y, theta, phi]
    x, y    : robot position (base_footprint)
    theta   : robot heading
    phi     : hitch angle (pivot_marker_drawbar_joint), positive = trailer left of robot

URDF-derived constants (do not change without updating the URDF):
    L1 = 0.5   m  — distance from robot rear axle to hitch point
                    (trailer_joint origin x = -0.5 from base_link)
    L2 = 2.0   m  — distance from hitch point to trailer axle centre
                    (drawbar 0.5 m + pivot offset 0.25 m + half container 1.5 m ≈ 2.0 m)
    PHI_MAX = 1.57 rad — joint limit from URDF
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from hybrid_a_star_planner.utils import Pos


# ---------------------------------------------------------------------------
# URDF-derived geometry (metres)
# ---------------------------------------------------------------------------
L1: float = 0.5          # robot rear axle  → hitch
L2: float = 2.0          # hitch            → trailer axle centre
PHI_MAX: float = 1.57    # max hitch angle  (URDF joint limit)


class MPC:
    def __init__(self, dt: float = 0.1, horizon: int = 10):
        self.dt = dt
        self.horizon = horizon

        # State weights: [x, y, theta, phi]
        self.Q  = np.diag([1.0, 1.0, 0.5, 2.0])   # phi gets extra weight
        self.R  = np.diag([0.1, 0.1])
        self.Qf = self.Q * 10

        self.v_max     =  1.0
        self.v_min     = -0.5
        self.omega_max =  1.0

        self.u_prev    = None
        self.obstacles = []

        # Latest hitch angle fed in from /joint_states
        self._phi: float = 0.0

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_physical_constraints(self, v_max: float, v_min: float, omega_max: float):
        self.v_max     = v_max
        self.v_min     = v_min
        self.omega_max = omega_max

    def set_weights(
        self,
        position_weight: float,
        heading_weight: float,
        control_weight: float,
        phi_weight: float = 2.0,
    ):
        self.Q  = np.diag([position_weight, position_weight, heading_weight, phi_weight])
        self.R  = np.diag([control_weight, control_weight])
        self.Qf = self.Q * 10

    def set_obstacles(self, obstacles: list):
        self.obstacles = obstacles

    def update_hitch_angle(self, phi: float):
        """Call this every control loop with the latest joint_states reading."""
        self._phi = float(phi)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_array(self, state) -> NDArray[np.float64]:
        """Accept Pos, list/array of length 3 or 4."""
        if hasattr(state, "x") and hasattr(state, "y") and hasattr(state, "theta"):
            return np.array([state.x, state.y, state.theta, 0.0])
        arr = np.asarray(state, dtype=float).flatten()
        if arr.size == 3:
            return np.append(arr, 0.0)   # pad phi = 0 for goal
        return arr[:4]

    def _to_pos(self, arr: NDArray[np.float64]) -> Pos:
        return Pos(x=arr[0], y=arr[1], theta=arr[2])

    # ------------------------------------------------------------------
    # Kinematic model  (unicycle tractor + passive trailer)
    #
    #   ẋ      = v · cos θ
    #   ẏ      = v · sin θ
    #   θ̇      = ω
    #   φ̇      = -v/L2 · sin φ  -  v·L1/(L1·L2) · cos φ · (L1·ω/v) …
    #
    # Exact discrete trailer kinematics (Altafini 1999):
    #   φ_next = φ  +  dt · [-v·sin(φ)/L2  -  ω·cos(φ)·L1/L2  +  ω]
    #   Simplified: φ̇ = ω − (v/L2)·sin(φ) − (v·L1/L2)·ω/v·cos(φ)
    #             = ω(1 − (L1/L2)·cos φ)  −  (v/L2)·sin φ
    # ------------------------------------------------------------------

    def robot_dynamics(
        self,
        state:   NDArray[np.float64],   # [x, y, theta, phi]
        control: NDArray[np.float64],   # [v, omega]
    ) -> NDArray[np.float64]:
        x, y, theta, phi = state
        v     = float(np.clip(control[0], self.v_min, self.v_max))
        omega = float(np.clip(control[1], -self.omega_max, self.omega_max))

        x_next     = x     + v * np.cos(theta) * self.dt
        y_next     = y     + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt

        # Trailer hitch angle kinematic equation
        phi_dot    = omega * (1.0 - (L1 / L2) * np.cos(phi)) - (v / L2) * np.sin(phi)
        phi_next   = phi + phi_dot * self.dt
        # Enforce joint limits
        phi_next   = float(np.clip(phi_next, -PHI_MAX, PHI_MAX))

        return np.array([x_next, y_next, theta_next, phi_next])

    # ------------------------------------------------------------------
    # Trajectory prediction
    # ------------------------------------------------------------------

    def predict_trajectory(
        self,
        state:         NDArray[np.float64],   # [x, y, theta, phi]
        controls_flat: NDArray[np.float64],   # interleaved [v0,w0, v1,w1, ...]
    ) -> NDArray[np.float64]:
        state    = np.asarray(state, dtype=float).flatten()[:4]
        controls = controls_flat.reshape(self.horizon, 2)
        traj     = np.zeros((self.horizon, 4))
        traj[0]  = state
        for i in range(1, self.horizon):
            traj[i] = self.robot_dynamics(traj[i - 1], controls[i - 1])
        return traj

    # ------------------------------------------------------------------
    # Collision cost
    # ------------------------------------------------------------------

    def compute_collision_cost(self, trajectory: NDArray[np.float64]) -> float:
        if not self.obstacles:
            return 0.0
        cost         = 0.0
        robot_radius = 0.25
        safety_margin = 0.1
        for state in trajectory:
            for ox, oy in self.obstacles:
                dist = np.hypot(state[0] - ox, state[1] - oy)
                if dist < robot_radius + safety_margin:
                    cost += 100.0 * (robot_radius + safety_margin - dist)
        return cost

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def cost_function(
        self,
        controls_flat:  NDArray[np.float64],
        current_state:  NDArray[np.float64],   # [x, y, theta, phi]
        reference:      NDArray[np.float64],   # [x, y, theta, phi=0]
    ) -> float:
        traj     = self.predict_trajectory(current_state, controls_flat)
        controls = controls_flat.reshape(self.horizon, 2)
        ref      = np.asarray(reference, dtype=float).flatten()[:4]

        cost = 0.0
        for i in range(self.horizon):
            error    = traj[i] - ref
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))  # wrap theta
            error[3] = np.arctan2(np.sin(error[3]), np.cos(error[3]))  # wrap phi
            cost    += error @ self.Q @ error + controls[i] @ self.R @ controls[i]

        final_error    = traj[-1] - ref
        final_error[2] = np.arctan2(np.sin(final_error[2]), np.cos(final_error[2]))
        final_error[3] = np.arctan2(np.sin(final_error[3]), np.cos(final_error[3]))
        cost += final_error @ self.Qf @ final_error
        cost += self.compute_collision_cost(traj)
        return cost

    # ------------------------------------------------------------------
    # Main solve entry-point
    # ------------------------------------------------------------------

    def solve_control(
        self,
        start_pos,   # Pos or [x,y,theta]  — current robot pose from odometry
        goal_pos,    # Pos or [x,y,theta]  — current waypoint
    ) -> tuple[NDArray[np.float64], list[Pos]]:
        """
        Returns
        -------
        optimal_controls : ndarray (horizon, 2)  — rows are [v, omega]
        predicted_traj   : list[Pos]             — for RViz visualisation
        """
        # Build 4-state vectors; inject live hitch angle into start
        start_arr      = self._to_array(start_pos)
        start_arr[3]   = self._phi          # ← real hitch angle from /joint_states
        goal_arr       = self._to_array(goal_pos)
        goal_arr[3]    = 0.0               # we always want phi → 0 at the goal

        # Warm start
        if self.u_prev is not None:
            u0 = np.vstack([self.u_prev[1:], self.u_prev[-1:]]).flatten()
        else:
            u0 = np.zeros(self.horizon * 2)

        # Interleaved bounds [v0,w0, v1,w1, ...]
        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.v_min, self.v_max))
            bounds.append((-self.omega_max, self.omega_max))

        result = minimize(
            fun     = lambda u: self.cost_function(u, start_arr, goal_arr),
            x0      = u0,
            method  = "SLSQP",
            bounds  = bounds,
            options = {"maxiter": 20, "ftol": 1e-3},
        )

        optimal_controls = result.x.reshape(self.horizon, 2)
        self.u_prev      = optimal_controls

        predicted_traj_arr = self.predict_trajectory(start_arr, result.x)
        predicted_traj     = [self._to_pos(s) for s in predicted_traj_arr]

        return optimal_controls, predicted_traj