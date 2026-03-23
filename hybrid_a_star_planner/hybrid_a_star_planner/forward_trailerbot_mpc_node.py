import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from hybrid_a_star_planner.forward_mpc import MPC
from hybrid_a_star_planner.a_star import AStar, Pos
from hybrid_a_star_planner.utils import quaternion_to_yaw


GOAL_TOLERANCE = 0.35  # Meters; consider goal reached if within this distance
GOAL_ANGLE_TOLERANCE = 0.15  # Radians; consider goal orientation reached if within this angle
MAX_SAFE_ARTICULATION = 0.95  # enter recovery earlier to avoid deep jackknife
RECOVERY_EXIT_ARTICULATION = 0.72  # stronger hysteresis before returning to MPC
RECOVERY_CHECK_PERIOD = 0.5  # seconds
RECOVERY_REQUIRED_DROP = 0.01  # rad in each check window
ARTICULATION_GUARD_THRESHOLD = 0.65  # bypass MPC when articulation is elevated
POST_DOCK_FORWARD_DISTANCE = 4.0  # meters to drive forward after reaching park pose
POST_FORWARD_EXTRA_DISTANCE = 3.0  # meters to drive after reaching forward staging goal

FORWARD_V_MAX = 0.25
FORWARD_V_MIN = 0.0
FORWARD_OMEGA_MAX = 0.25
FORWARD_POSITION_WEIGHT = 10.0
FORWARD_HEADING_WEIGHT = 2.0
FORWARD_CONTROL_WEIGHT = 3.0
FORWARD_PHI_WEIGHT = 2.0

REVERSE_V_MAX = -0.05
REVERSE_V_MIN = -0.2
REVERSE_OMEGA_MAX = 0.25
REVERSE_POSITION_WEIGHT = 0.5
REVERSE_HEADING_WEIGHT = 1.0
REVERSE_CONTROL_WEIGHT = 0.5
REVERSE_PHI_WEIGHT = 1.0


class ForwardTrailerBotMPCNode(Node, AStar):
    def __init__(self, mpc_controller: MPC):
        Node.__init__(self, "forward_trailerbot_mpc_node")
        AStar.__init__(self)
        self.mpc = mpc_controller
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Works with both Reliable and Best Effort
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.path_pub = self.create_publisher(Path, "trajectory", 10)
        self.mpc_path_pub = self.create_publisher(Path, "mpc_trajectory", 10)

        self.start_pos_sub = self.create_subscription(
            PoseWithCovarianceStamped, "start_pos", self.start_pos_cb, 1
        )
        self.goal_pos_sub = self.create_subscription(
            PoseStamped, "goal_pos", self.goal_pos_cb, 1
        )
        self.clicked_pt = self.create_subscription(
            PoseWithCovarianceStamped, "park_pos", self.clicked_point_cb, 1
        )

        self.path = Path()
        self.path.header.frame_id = "odom"

        self.mpc_path = Path()
        self.mpc_path.header.frame_id = "odom"

        self.path_index = 1
        self.traj: list[Pos] = []

        self.current_pose = None
        self.goal_pose = None # (3.0, 0.0, 0.0, 0.0)  # [x, y, theta, phi]
        self._phi = 0.0  # Hitch angle
        self._recovering_articulation = False
        self._last_recovery_log_time = 0.0
        self._recovery_omega_sign = 1.0
        self._recovery_last_phi_abs = 0.0
        self._recovery_last_eval_time = 0.0
        self._reverse_phase_started = False
        self._post_forward_extension_started = False

        # Run control loop at the same dt as your MPC
        self.timer = self.create_timer(self.mpc.dt, self.control_loop)
        self.start_pos = Pos.default()
        self.goal_pos = Pos.default()
        self.park_pos = Pos.default()
        self.get_logger().info("TurtleBot MPC Node initialized.")

    def _update_goal_from_path(self):
        if not self.traj or self.current_pose is None:
            return

        cx, cy = self.current_pose[0], self.current_pose[1]

        # Search around the current progress index so the tracker can recover
        # if the robot drifts, while still preserving forward progress.
        look_back = 4
        look_ahead = 12
        start_idx = max(0, self.path_index - look_back)
        end_idx = min(len(self.traj), self.path_index + look_ahead)
        search_range = range(start_idx, end_idx)

        nearest_idx = self.path_index
        min_dist = float("inf")
        for i in search_range:
            d = np.hypot(self.traj[i].x - cx, self.traj[i].y - cy)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        # If we drifted far, do a full re-acquire to avoid chasing stale targets.
        if min_dist > 1.2:
            nearest_idx = min(
                range(len(self.traj)),
                key=lambda i: np.hypot(self.traj[i].x - cx, self.traj[i].y - cy),
            )

        # Keep lookahead short when articulation is high.
        if abs(self._phi) > 0.9:
            lookahead = 0
        elif abs(self._phi) > 0.6:
            lookahead = 1
        else:
            lookahead = 3
        self.path_index = min(nearest_idx + lookahead, len(self.traj) - 1)

        target = self.traj[self.path_index]
        self.goal_pose = [
            target.x,
            target.y,
            target.theta,
            self._target_phi_for_index(self.path_index),
        ]

    def _target_phi_for_index(self, index: int) -> float:
        # Encourage articulation decay throughout the path and enforce
        # straight trailer near final docking section.
        if index >= max(0, len(self.traj) - 8):
            return 0.0
        if abs(self._phi) > 0.7:
            return 0.0
        return float(np.clip(0.4 * self._phi, -0.35, 0.35))

    def _publish_path_and_set_goal(self):
        self.path.poses.clear()  # type: ignore
        for pos in self.traj:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "odom"
            pose_stamped.pose.position.x = pos.x
            pose_stamped.pose.position.y = pos.y
            qz = np.sin(pos.theta / 2)
            qw = np.cos(pos.theta / 2)
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw
            self.path.poses.append(pose_stamped)  # type: ignore
        self.path_pub.publish(self.path)

        self.path_index = 1 if len(self.traj) > 1 else 0
        self.goal_pose = [
            self.traj[self.path_index].x,
            self.traj[self.path_index].y,
            self.traj[self.path_index].theta,
            self._target_phi_for_index(self.path_index),
        ]

    def _start_reverse_phase(self):
        if self.current_pose is None:
            return

        self.get_logger().info("Forward goal reached. Switching MPC to reverse docking mode.")
        self._reverse_phase_started = True
        self._recovering_articulation = False

        self.mpc.set_physical_constraints(
            v_max=REVERSE_V_MAX,
            v_min=REVERSE_V_MIN,
            omega_max=REVERSE_OMEGA_MAX,
        )
        self.mpc.set_weights(
            position_weight=REVERSE_POSITION_WEIGHT,
            heading_weight=REVERSE_HEADING_WEIGHT,
            control_weight=REVERSE_CONTROL_WEIGHT,
            phi_weight=REVERSE_PHI_WEIGHT,
        )
        self.mpc.u_prev = None
        self.mpc.set_last_applied_control(0.0, 0.0)

        start_for_plan = Pos(
            x=self.current_pose[0],
            y=self.current_pose[1],
            theta=self.current_pose[2],
            phi=self._phi,
        )

        self.traj = self.plan(start_for_plan, self.park_pos)
        if not self.traj:
            # Fallback: force a direct segment so reverse phase can still execute.
            self.get_logger().warn("A* reverse plan failed. Using direct fallback reverse segment.")
            self.traj = [
                Pos(
                    x=start_for_plan.x,
                    y=start_for_plan.y,
                    theta=start_for_plan.theta,
                    phi=start_for_plan.phi,
                ),
                Pos(
                    x=self.park_pos.x,
                    y=self.park_pos.y,
                    theta=self.park_pos.theta,
                    phi=0.0,
                ),
            ]

        # Ensure terminal park point exists for robust goal detection.
        if not self.traj or (
            np.hypot(self.traj[-1].x - self.park_pos.x, self.traj[-1].y - self.park_pos.y) > 1e-3
        ):
            self.traj.append(
                Pos(
                    x=self.park_pos.x,
                    y=self.park_pos.y,
                    theta=self.park_pos.theta,
                    phi=0.0,
                )
            )

        self.get_logger().info(f"Reverse path has {len(self.traj)} waypoints.")
        self._publish_path_and_set_goal()

    def _publish_recovery_cmd(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        phi_abs = abs(self._phi)

        # Flip steering direction if articulation is not reducing enough.
        if now_sec - self._recovery_last_eval_time >= RECOVERY_CHECK_PERIOD:
            if phi_abs > self._recovery_last_phi_abs - RECOVERY_REQUIRED_DROP:
                self._recovery_omega_sign *= -1.0
            self._recovery_last_phi_abs = phi_abs
            self._recovery_last_eval_time = now_sec

        # Keep a small crawl; use reverse crawl in reverse docking phase.
        if phi_abs > 1.2:
            v_mag = 0.03
            omega_gain = 0.20
        else:
            v_mag = 0.05
            omega_gain = 0.16

        v = -v_mag if self._reverse_phase_started else v_mag
        omega = float(np.clip(omega_gain * self._recovery_omega_sign, -0.20, 0.20))
        self._publish_cmd(v, omega)
        self.mpc.set_last_applied_control(float(v), float(omega))

        if now_sec - self._last_recovery_log_time > 1.0:
            self.get_logger().warn(
                f"Articulation recovery active (phi={self._phi:.2f} rad, v={v:.2f}, omega={omega:.2f}, dir={self._recovery_omega_sign:+.0f})."
            )
            self._last_recovery_log_time = now_sec

    def _publish_articulation_guard_cmd(self):
        # Stay out of optimizer saturation: unwind articulation with gentle motion.
        if self.goal_pose is None or self.current_pose is None:
            return

        heading_error = self.goal_pose[2] - self.current_pose[2]
        heading_error = float(np.arctan2(np.sin(heading_error), np.cos(heading_error)))

        v_mag = 0.04 if abs(self._phi) > 0.8 else 0.05
        v = -v_mag if self._reverse_phase_started else v_mag
        steer_unwind = -0.18 * np.sign(self._phi)
        steer_heading = 0.05 * heading_error
        omega = float(np.clip(steer_unwind + steer_heading, -0.14, 0.14))

        self._publish_cmd(v, omega)
        self.mpc.set_last_applied_control(float(v), float(omega))

    def odom_callback(self, msg: Odometry):
        # Change to info so it shows up by default
        # self.get_logger().info("Received odometry message.")

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # Quaternion to Euler (Yaw)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        self.current_pose = [pos.x, pos.y, yaw, self._phi]
        # self.get_logger().info(f"Current pose: {self.current_pose}")

    def _joint_cb(self, msg: JointState):
        if "pivot_marker_drawbar_joint" in msg.name:
            idx = list(msg.name).index("pivot_marker_drawbar_joint")
            self._phi = msg.position[idx]
            # self.get_logger().info(f"Updated hitch angle: {self.hitch_angle:.2f}")

    def start_pos_cb(self, msg: PoseWithCovarianceStamped):
        self.start_pos.x = msg.pose.pose.position.x
        self.start_pos.y = msg.pose.pose.position.y
        self.start_pos.theta = quaternion_to_yaw(msg.pose.pose.orientation)
        self.get_logger().info(
            f"Received start: x={self.start_pos.x:.2f}, y={self.start_pos.y:.2f}, theta={self.start_pos.theta:.2f}"
        )

    def clicked_point_cb(self, msg: PoseWithCovarianceStamped):
        self.get_logger().info(
            f"Received clicked point: x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}"
        )
        self.park_pos.x = msg.pose.pose.position.x
        self.park_pos.y = msg.pose.pose.position.y
        self.park_pos.theta = quaternion_to_yaw(msg.pose.pose.orientation)
        self.park_pos.phi = 0.0

    def goal_pos_cb(self, msg: PoseStamped):
        self._reverse_phase_started = False
        self._recovering_articulation = False
        self._post_forward_extension_started = False

        self.mpc.set_physical_constraints(
            v_max=FORWARD_V_MAX,
            v_min=FORWARD_V_MIN,
            omega_max=FORWARD_OMEGA_MAX,
        )
        self.mpc.set_weights(
            position_weight=FORWARD_POSITION_WEIGHT,
            heading_weight=FORWARD_HEADING_WEIGHT,
            control_weight=FORWARD_CONTROL_WEIGHT,
            phi_weight=FORWARD_PHI_WEIGHT,
        )
        self.mpc.u_prev = None

        self.park_pos.x = msg.pose.position.x
        self.park_pos.y = msg.pose.position.y
        self.park_pos.theta = quaternion_to_yaw(msg.pose.orientation)
        self.park_pos.phi = 0.0

        # Phase 1 goal: move to a fixed world-frame +X staging point, e.g.
        # goal (2,2) -> staging (6,2), then reverse back to (2,2).
        self.goal_pos.x = self.park_pos.x + POST_DOCK_FORWARD_DISTANCE
        self.goal_pos.y = self.park_pos.y
        self.goal_pos.theta = 0.0
        self.goal_pos.phi = 0.0
        self.get_logger().info(
            f"Received park goal: x={self.park_pos.x:.2f}, y={self.park_pos.y:.2f}, theta={self.park_pos.theta:.2f}"
        )
        self.get_logger().info(
            f"Forward staging goal (+X): x={self.goal_pos.x:.2f}, y={self.goal_pos.y:.2f}, theta={self.goal_pos.theta:.2f}"
        )

        start_for_plan = self.start_pos
        if self.current_pose is not None:
            start_for_plan = Pos(
                x=self.current_pose[0],
                y=self.current_pose[1],
                theta=self.current_pose[2],
                phi=self._phi,
            )

        self.traj = self.plan(start_for_plan, self.goal_pos)
        self.get_logger().info(f"Initial A* path has {len(self.traj)} waypoints.")

        if self.traj:
            self.get_logger().info(f"Planned path with {len(self.traj)} waypoints.")
            self._publish_path_and_set_goal()
        else:
            self.get_logger().warn("Failed to plan path with A*.")

    def control_loop(self):
        if self.current_pose is None or self.goal_pose is None or not self.traj:
            return

        # Enter/exit recovery mode using hysteresis.
        was_recovering = self._recovering_articulation
        phi_abs = abs(self._phi)
        if phi_abs > MAX_SAFE_ARTICULATION:
            self._recovering_articulation = True
        elif self._recovering_articulation and phi_abs < RECOVERY_EXIT_ARTICULATION:
            self._recovering_articulation = False

        if self._recovering_articulation and not was_recovering:
            self._recovery_omega_sign = float(np.sign(self._phi)) if abs(self._phi) > 1e-3 else 1.0
            self._recovery_last_phi_abs = phi_abs
            self._recovery_last_eval_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().warn(
                f"Entering articulation recovery (phi={self._phi:.2f} rad)."
            )

        # In recovery mode, temporarily bypass MPC and unwind articulation.
        if self._recovering_articulation:
            self._publish_recovery_cmd()
            return

        # Use the smart update to find the goal waypoint
        self._update_goal_from_path()

        # Calculate distance to final waypoint
        dx = self.traj[-1].x - self.current_pose[0]
        dy = self.traj[-1].y - self.current_pose[1]
        final_dist = np.hypot(dx, dy)

        if final_dist < GOAL_TOLERANCE:
            if not self._post_forward_extension_started:
                self.get_logger().info(
                    "Forward phase goal reached. Extending +3.0m before shutdown."
                )
                self._post_forward_extension_started = True
                self._publish_cmd(0.0, 0.0)

                extension_target = Pos(
                    x=self.current_pose[0] + POST_FORWARD_EXTRA_DISTANCE,
                    y=self.current_pose[1],
                    theta=self.current_pose[2],
                    phi=0.0,
                )
                self.traj = [
                    Pos(
                        x=self.current_pose[0],
                        y=self.current_pose[1],
                        theta=self.current_pose[2],
                        phi=self._phi,
                    ),
                    extension_target,
                ]
                self._publish_path_and_set_goal()
            else:
                self.get_logger().info(
                    "Post-forward extension reached. Stopping and exiting."
                )
                self._publish_cmd(0.0, 0.0)
                self.goal_pose = None
                rclpy.shutdown()
            return

        # Bypass MPC when articulation is elevated to avoid repeated -omega_max saturation.
        if abs(self._phi) > ARTICULATION_GUARD_THRESHOLD:
            self._publish_articulation_guard_cmd()
            return

        self.get_logger().info(
            f"Current pose: {self.current_pose}, Goal pose: {self.goal_pose}"
        )
        # Solve MPC
        optimal_controls = self.mpc.solve_control(self.current_pose, self.goal_pose)
        v, omega = optimal_controls[0]
        self.get_logger().info(f"Optimal control: v={v:.2f}, omega={omega:.2f}")

        # --- STABILIZATION: Slow down for the trailer ---
        # If the trailer is at a sharp angle (> 25 deg), slow down the robot
        if abs(self._phi) > 0.45:
            if self._reverse_phase_started:
                v = float(np.clip(v, -0.15, 0.0))
            else:
                v = float(np.clip(v, 0.0, 0.15))

        # Aggressive protection when articulation is already high.
        if abs(self._phi) > 0.6:
            if self._reverse_phase_started:
                v = float(np.clip(v, -0.06, 0.0))
            else:
                v = float(np.clip(v, 0.0, 0.06))
            omega = float(np.clip(omega, -0.06, 0.06))
        
        # If the robot is steering hard, slow down
        if abs(omega) > 0.15:
            if self._reverse_phase_started:
                v = float(np.clip(v, -0.15, 0.0))
            else:
                v = float(np.clip(v, 0.0, 0.15))

        # When articulation is elevated, limit steering aggressiveness too.
        if abs(self._phi) > 0.6:
            omega = float(np.clip(omega, -0.18, 0.18))

        self._publish_cmd(v, omega)
        self.mpc.set_last_applied_control(float(v), float(omega))

    # def control_loop(self):
    #     if self.current_pose is None or self.goal_pose is None:
    #         self.get_logger().warn(
    #             "Waiting for current pose and goal pose to be set..."
    #         )
    #         return

    #     # Refresh target from trajectory using nearest + lookahead indexing.
    #     # self._update_goal_from_path()

    #     self.goal_pose = [self.traj[self.path_index].x, self.traj[self.path_index].y, self.traj[self.path_index].theta, self._target_phi_for_index(self.path_index)]

    #     # check if goal is reached
    #     dx = self.goal_pose[0] - self.current_pose[0]
    #     dy = self.goal_pose[1] - self.current_pose[1]
    #     distance_to_goal = np.hypot(dx, dy)
    #     if distance_to_goal < GOAL_TOLERANCE and abs(wrap_angle(self.goal_pose[2] - self.current_pose[2])) < GOAL_ANGLE_TOLERANCE:
    #         self.get_logger().info("Goal reached!")
    #         # self._publish_cmd(0.0, 0.0)  # Stop the robot
    #         # exit(0)
    #         # return
    #         if self.path_index >= len(self.traj) - 1:
    #             self.get_logger().info("Final goal reached!")
    #             cmd = Twist()
    #             cmd.linear.x = 0.0
    #             cmd.angular.z = 0.0
    #             self.cmd_pub.publish(cmd)
    #             return
    #         self.path_index += 1
    #         if self.path_index < len(self.path.poses):
    #             self.goal_pose = [
    #                 self.traj[self.path_index].x,
    #                 self.traj[self.path_index].y,
    #                 self.traj[self.path_index].theta,
    #                 self._target_phi_for_index(self.path_index),
    #             ]

    #     # 1. Solve MPC for the current state
    #     # Note: Modify your solve() to return 'optimal_controls' instead of just the trajectory
    #     optimal_controls = self.mpc.solve_control(self.current_pose, self.goal_pose)
    #     self.get_logger().info(
    #         f"Current pose: {self.current_pose}, Goal pose: {self.goal_pose}"
    #     )

    #     # 2. Extract the first control command (v, omega)
    #     v, omega = optimal_controls[0]

    #     """
    #     # If heading error is large, slow down linear speed so the robot turns first.
    #     heading_error = wrap_angle(self.goal_pose[2] - self.current_pose[2])
    #     if abs(heading_error) > 0.7:
    #         v = float(np.clip(v, 0.03, 0.10))

    #     # Avoid reverse oscillation during early/mid path tracking.
    #     if self.path_index < max(1, len(self.traj) - 6):
    #         v = float(max(v, 0.0))

    #     if distance_to_goal < 0.6:
    #         omega = float(np.clip(omega, -0.25, 0.25))
    #     """
    #     # --- DYNAMIC STABILIZATION LOGIC ---
    #     # A) Slow down if the hitch angle is getting steep
    #     # phi_abs = abs(self._phi)
    #     # if phi_abs > 0.4: # Starting to turn
    #     #     # Reduce velocity proportionally to the hitch angle
    #     #     v = v * np.clip(1.0 - (phi_abs - 0.4), 0.2, 1.0)
        
    #     # B) Slow down if MPC is commanding maximum steering
    #     # if abs(omega) > 0.2:
    #     #     v = v * 0.5 

    #     # # C) Ensure we don't stall
    #     # if distance_to_goal > 0.1:
    #     #     v = max(v, 0.05)
    #     # -----------------------------------

    #     self.get_logger().info(f"Optimal control: v={v:.2f}, omega={omega:.2f}")

    #     # 3. Publish to TurtleBot
    #     self._publish_cmd(v, omega)
    #     self.mpc.set_last_applied_control(float(v), float(omega))

    def _publish_cmd(self, v: float, omega: float):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    # Initialize your MPC class from your code
    my_mpc = MPC(dt=0.1, horizon=25)
    my_mpc.set_physical_constraints(
        v_max=FORWARD_V_MAX, v_min=FORWARD_V_MIN, omega_max=FORWARD_OMEGA_MAX
    )
    # my_mpc.set_weights(
    #     position_weight=2.0,   # Let it drift slightly to maintain smoothness
    #     heading_weight=1.0, 
    #     control_weight=5.0,    # INCREASE: This is the most important change. 
    #                         # It forces the steering to be MUCH slower.
    #     phi_weight=1.0         # DECREASE: Stop fighting the planner's hitch angle.
    # )
    my_mpc.set_weights(
        position_weight=FORWARD_POSITION_WEIGHT,
        heading_weight=FORWARD_HEADING_WEIGHT,
        control_weight=FORWARD_CONTROL_WEIGHT,
        phi_weight=FORWARD_PHI_WEIGHT,
    )
    node = ForwardTrailerBotMPCNode(my_mpc)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
