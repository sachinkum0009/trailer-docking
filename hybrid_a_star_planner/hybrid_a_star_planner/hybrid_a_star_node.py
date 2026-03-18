#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, Quaternion
from geometry_msgs.msg import Twist
from hybrid_a_star_planner.mpc import MPC
from hybrid_a_star_planner.utils import Pos

import math


def quaternion_to_yaw(q: Quaternion) -> float:
    """Convert quaternion to yaw angle."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class HybridAStarNode(Node, MPC):
    def __init__(self):
        Node.__init__(self, "hybrid_a_star")
        MPC.__init__(self, dt=0.1, horizon=10)

        self.start_pos_sub = self.create_subscription(
            PoseWithCovarianceStamped, "start_pos", self.start_pos_cb, 1
        )
        self.goal_pos_sub = self.create_subscription(
            PoseStamped, "goal_pos", self.goal_pos_cb, 1
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        self.path_pub = self.create_publisher(Path, "trajectory", 10)
        self.mpc_path_pub = self.create_publisher(Path, "mpc_trajectory", 10)

        self.path = Path()
        self.path.header.frame_id = "map"

        self.mpc_path = Path()
        self.mpc_path.header.frame_id = "map"

        self.start_pos = Pos.default()
        self.goal_pos = Pos.default()

        self.get_logger().info("Hybrid A* node started.")

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

        mpc_trajectory = self.solve(self.start_pos, self.goal_pos)
        self.get_logger().info(f"MPC trajectory computed: {len(mpc_trajectory)} points")

        self.mpc_path.poses.clear()
        for pos in mpc_trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = pos.x
            pose_stamped.pose.position.y = pos.y
            self.mpc_path.poses.append(pose_stamped)
        self.mpc_path_pub.publish(self.mpc_path)

        cmd_vel = Twist()
        if len(mpc_trajectory) > 1:
            dx = mpc_trajectory[1].x - mpc_trajectory[0].x
            dy = mpc_trajectory[1].y - mpc_trajectory[0].y
            cmd_vel.linear.x = math.sqrt(dx * dx + dy * dy) / self.dt
            cmd_vel.angular.z = (
                mpc_trajectory[1].theta - mpc_trajectory[0].theta
            ) / self.dt

        cmd_vel.linear.x = max(self.v_min, min(self.v_max, cmd_vel.linear.x))
        cmd_vel.angular.z = max(-self.omega_max, min(self.omega_max, cmd_vel.angular.z))

        self.cmd_vel_pub.publish(cmd_vel)
        self.get_logger().info(
            f"Published cmd_vel: v={cmd_vel.linear.x:.2f}, omega={cmd_vel.angular.z:.2f}"
        )


def main():
    rclpy.init()
    node = HybridAStarNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
