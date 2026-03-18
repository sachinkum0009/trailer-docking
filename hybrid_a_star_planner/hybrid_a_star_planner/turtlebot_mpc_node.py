import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped, Quaternion
from nav_msgs.msg import Odometry, Path
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from hybrid_a_star_planner.mpc import MPC
from hybrid_a_star_planner.a_star import AStar, Pos

def quaternion_to_yaw(q: Quaternion) -> float:
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

class TurtleBotMPCNode(Node, AStar):
    def __init__(self, mpc_controller):
        Node.__init__(self, "turtlebot_mpc_node")
        AStar.__init__(self)
        self.mpc = mpc_controller
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Works with both Reliable and Best Effort
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        # self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile
        )
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

        self.path_index = 0
        self.traj: list[Pos] = []

        self.current_pose = None
        self.goal_pose = None # [0.0, 0.0, 0.0]  # Your target

        # Run control loop at the same dt as your MPC
        self.timer = self.create_timer(self.mpc.dt, self.control_loop)
        self.start_pos = Pos.default()
        self.goal_pos = Pos.default()
        self.get_logger().info("TurtleBot MPC Node initialized.")

    def odom_callback(self, msg: Odometry):
        # Change to info so it shows up by default
        # self.get_logger().info("Received odometry message.")

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation

        # Quaternion to Euler (Yaw)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        self.current_pose = [pos.x, pos.y, yaw]
        # self.get_logger().info(f"Current pose: {self.current_pose}")

    # def odom_callback(self, msg: Odometry):
    #     self.get_logger().debug("Received odometry message.")
    #     # Extract x, y, and convert quaternion to yaw (theta)
    #     # pos = msg.pose.pose.position
    #     # # Simplified quaternion to yaw conversion
    #     # q = msg.pose.pose.orientation
    #     # siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    #     # cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    #     # yaw = np.arctan2(siny_cosp, cosy_cosp)
    #     # self.current_pose = [pos.x, pos.y, yaw]
    #     # self.get_logger().debug(f"Current pose updated: {self.current_pose}")

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
        self.get_logger().info(
            f"Received goal: x={self.goal_pos.x:.2f}, y={self.goal_pos.y:.2f}, theta={self.goal_pos.theta:.2f}"
        )

        self.traj = self.plan(self.start_pos, self.goal_pos)
        if self.traj:
            self.get_logger().info(f"Planned path with {len(self.traj)} waypoints.")
            self.path.poses.clear() #type: ignore
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
                self.path.poses.append(pose_stamped) # type: ignore
            self.path_pub.publish(self.path)
            self.goal_pose = [self.traj[0].x, self.traj[0].y, self.traj[0].theta]
        else:
            self.get_logger().warn("Failed to plan path with A*.")

    def control_loop(self):
        if self.current_pose is None or self.goal_pose is None:
            self.get_logger().warn("Waiting for current pose and goal pose to be set...")
            return

        # check if goal is reached
        dx = self.goal_pose[0] - self.current_pose[0]
        dy = self.goal_pose[1] - self.current_pose[1]
        distance_to_goal = np.hypot(dx, dy)
        if distance_to_goal < 0.1:
            self.get_logger().info("Goal reached!")
            # exit(0)
            # return
            if self.path_index >= len(self.traj) - 1:
                self.get_logger().info("Final goal reached!")
                cmd = TwistStamped()
                cmd.header.stamp = self.get_clock().now().to_msg()
                self.cmd_pub.publish(cmd)
                exit(0)
                return
            self.path_index += 1
            if self.path_index < len(self.path.poses):
                self.goal_pose = [self.traj[self.path_index].x, self.traj[self.path_index].y, self.traj[self.path_index].theta]

        # 1. Solve MPC for the current state
        # Note: Modify your solve() to return 'optimal_controls' instead of just the trajectory
        optimal_controls, predicted_traj = self.mpc.solve_control(
            self.current_pose, self.goal_pose
        )
        self.get_logger().info(f"Current pose: {self.current_pose}, Goal pose: {self.goal_pose}")

        # 2. Extract the first control command (v, omega)
        v, omega = optimal_controls[0]
        self.get_logger().info(f"Optimal control: v={v:.2f}, omega={omega:.2f}")

        # 3. Publish to TurtleBot
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(omega)
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    # Initialize your MPC class from your code
    my_mpc = MPC(dt=0.1, horizon=40)
    my_mpc.set_physical_constraints(v_max=0.25, v_min=-0.25, omega_max=0.5)
    my_mpc.set_weights(position_weight=1.0, heading_weight=0.5, control_weight=0.1)
    node = TurtleBotMPCNode(my_mpc)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
