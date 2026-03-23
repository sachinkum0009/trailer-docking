import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Quaternion
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from hybrid_a_star_planner.forward_mpc import MPC
from hybrid_a_star_planner.a_star import AStar, Pos
from hybrid_a_star_planner.utils import quaternion_to_yaw, wrap_angle


GOAL_TOLERANCE = 0.15  # Meters; consider goal reached if within this distance


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

        self.path = Path()
        self.path.header.frame_id = "odom"

        self.mpc_path = Path()
        self.mpc_path.header.frame_id = "odom"

        self.path_index = 1
        self.traj: list[Pos] = []

        self.current_pose = None
        self.goal_pose = (3.0, 0.0, 0.0, 0.0)  # [x, y, theta, phi]
        self._phi = 0.0  # Hitch angle

        # Run control loop at the same dt as your MPC
        self.timer = self.create_timer(self.mpc.dt, self.control_loop)
        self.start_pos = Pos.default()
        self.goal_pos = Pos.default()
        self.get_logger().info("TurtleBot MPC Node initialized.")

    def _update_goal_from_path(self):
        if not self.traj or self.current_pose is None:
            return

        cx, cy = self.current_pose[0], self.current_pose[1]
        start_idx = max(0, min(self.path_index, len(self.traj) - 1))

        nearest_idx = start_idx
        nearest_d2 = float("inf")
        for i in range(start_idx, len(self.traj)):
            dx = self.traj[i].x - cx
            dy = self.traj[i].y - cy
            d2 = dx * dx + dy * dy
            if d2 < nearest_d2:
                nearest_d2 = d2
                nearest_idx = i

        lookahead = 3
        target_idx = min(nearest_idx + lookahead, len(self.traj) - 1)
        self.path_index = target_idx
        self.goal_pose = [
            self.traj[target_idx].x,
            self.traj[target_idx].y,
            self.traj[target_idx].theta,
            self._target_phi_for_index(target_idx),
        ]

    def _target_phi_for_index(self, index: int) -> float:
        # Keep articulation objective soft for intermediate waypoints;
        # enforce straight trailer at final docking pose.
        if index >= len(self.traj) - 1:
            return 0.0
        return float(self._phi)

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

    def goal_pos_cb(self, msg: PoseStamped):
        self.goal_pos.x = msg.pose.position.x
        self.goal_pos.y = msg.pose.position.y
        self.goal_pos.theta = quaternion_to_yaw(msg.pose.orientation)
        self.goal_pos.phi = 0.0
        self.get_logger().info(
            f"Received goal: x={self.goal_pos.x:.2f}, y={self.goal_pos.y:.2f}, theta={self.goal_pos.theta:.2f}"
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
        if self.traj:
            self.get_logger().info(f"Planned path with {len(self.traj)} waypoints.")
            self.path.poses.clear()  # type: ignore
            for pos in self.traj:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "odom"
                pose_stamped.pose.position.x = pos.x
                pose_stamped.pose.position.y = pos.y
                # Convert theta back to quaternion for visualization
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
        else:
            self.get_logger().warn("Failed to plan path with A*.")

    def control_loop(self):
        if self.current_pose is None or self.goal_pose is None:
            self.get_logger().warn(
                "Waiting for current pose and goal pose to be set..."
            )
            return

        # Refresh target from trajectory using nearest + lookahead indexing.
        # self._update_goal_from_path()

        # check if goal is reached
        dx = self.goal_pose[0] - self.current_pose[0]
        dy = self.goal_pose[1] - self.current_pose[1]
        distance_to_goal = np.hypot(dx, dy)
        if distance_to_goal < GOAL_TOLERANCE:
            self.get_logger().info("Goal reached!")
            self._publish_cmd(0.0, 0.0)  # Stop the robot
            exit(0)
            return
            # if self.path_index >= len(self.traj) - 1:
            #     self.get_logger().info("Final goal reached!")
            #     cmd = Twist()
            #     cmd.linear.x = 0.0
            #     cmd.angular.z = 0.0
            #     self.cmd_pub.publish(cmd)
            #     return
            # self.path_index += 1
            # if self.path_index < len(self.path.poses):
            #     self.goal_pose = [
            #         self.traj[self.path_index].x,
            #         self.traj[self.path_index].y,
            #         self.traj[self.path_index].theta,
            #         self._target_phi_for_index(self.path_index),
            #     ]

        # 1. Solve MPC for the current state
        # Note: Modify your solve() to return 'optimal_controls' instead of just the trajectory
        optimal_controls = self.mpc.solve_control(self.current_pose, self.goal_pose)
        self.get_logger().info(
            f"Current pose: {self.current_pose}, Goal pose: {self.goal_pose}"
        )

        # 2. Extract the first control command (v, omega)
        v, omega = optimal_controls[0]

        """
        # If heading error is large, slow down linear speed so the robot turns first.
        heading_error = wrap_angle(self.goal_pose[2] - self.current_pose[2])
        if abs(heading_error) > 0.7:
            v = float(np.clip(v, 0.03, 0.10))

        # Avoid reverse oscillation during early/mid path tracking.
        if self.path_index < max(1, len(self.traj) - 6):
            v = float(max(v, 0.0))

        if distance_to_goal < 0.6:
            omega = float(np.clip(omega, -0.25, 0.25))
        """

        self.get_logger().info(f"Optimal control: v={v:.2f}, omega={omega:.2f}")

        # 3. Publish to TurtleBot
        self._publish_cmd(v, omega)
        self.mpc.set_last_applied_control(float(v), float(omega))

    def _publish_cmd(self, v: float, omega: float):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    # Initialize your MPC class from your code
    my_mpc = MPC(dt=0.1, horizon=18)
    my_mpc.set_physical_constraints(v_max=0.25, v_min=-0.25, omega_max=0.5)
    my_mpc.set_weights(
        position_weight=1.0, heading_weight=0.5, control_weight=0.5, phi_weight=0.1
    )
    node = ForwardTrailerBotMPCNode(my_mpc)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
