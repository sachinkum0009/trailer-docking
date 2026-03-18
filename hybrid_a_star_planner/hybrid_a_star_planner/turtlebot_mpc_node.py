#!/usr/bin/env python3
"""
TurtleBot MPC Node — trailer-aware version.

Subscribes to /joint_states to read the live hitch angle
(pivot_marker_drawbar_joint) and feeds it into the MPC at every control step.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Quaternion
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from hybrid_a_star_planner.mpc import MPC
from hybrid_a_star_planner.a_star import AStar, Pos


HITCH_JOINT_NAME = "pivot_marker_drawbar_joint"


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class TurtleBotMPCNode(Node, AStar):
    def __init__(self, mpc_controller: MPC):
        Node.__init__(self, "turtlebot_mpc_node")
        AStar.__init__(self)
        self.mpc = mpc_controller

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Publishers ────────────────────────────────────────────────
        # trailerbot uses plain Twist on /cmd_vel (DiffDrive plugin)
        self.cmd_pub      = self.create_publisher(Twist,  "/cmd_vel",       10)
        self.path_pub     = self.create_publisher(Path,   "trajectory",     10)
        self.mpc_path_pub = self.create_publisher(Path,   "mpc_trajectory", 10)

        # ── Subscribers ───────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, qos_best_effort
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10
        )
        self.start_pos_sub = self.create_subscription(
            PoseWithCovarianceStamped, "start_pos", self._start_pos_cb, 1
        )
        self.goal_pos_sub = self.create_subscription(
            PoseStamped, "goal_pos", self._goal_pos_cb, 1
        )

        # ── State ─────────────────────────────────────────────────────
        self.current_pose: list[float] | None = None   # [x, y, theta]
        self.goal_pose:    list[float] | None = None   # [x, y, theta]
        self.hitch_angle:  float               = 0.0   # phi from joint_states

        self.path        = Path()
        self.path.header.frame_id = "odom"
        self.mpc_path    = Path()
        self.mpc_path.header.frame_id = "odom"

        self.traj:        list[Pos] = []
        self.path_index:  int       = 0
        self.start_pos    = Pos.default()
        self.goal_pos     = Pos.default()

        # ── Control timer ─────────────────────────────────────────────
        self.timer = self.create_timer(self.mpc.dt, self._control_loop)
        self.get_logger().info("TurtleBot MPC Node (trailer-aware) initialised.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        pos = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        self.current_pose = [pos.x, pos.y, float(yaw)]

    def _joint_states_cb(self, msg: JointState):
        """Extract hitch angle and push it into the MPC."""
        if HITCH_JOINT_NAME in msg.name:
            idx = msg.name.index(HITCH_JOINT_NAME)
            self.hitch_angle = float(msg.position[idx])
            self.mpc.update_hitch_angle(self.hitch_angle)

    def _start_pos_cb(self, msg: PoseWithCovarianceStamped):
        self.start_pos.x     = msg.pose.pose.position.x
        self.start_pos.y     = msg.pose.pose.position.y
        self.start_pos.theta = quaternion_to_yaw(msg.pose.pose.orientation)
        self.get_logger().info(
            f"Start set: x={self.start_pos.x:.2f}  y={self.start_pos.y:.2f}"
            f"  θ={self.start_pos.theta:.2f}"
        )

    def _goal_pos_cb(self, msg: PoseStamped):
        self.goal_pos.x     = msg.pose.position.x
        self.goal_pos.y     = msg.pose.position.y
        self.goal_pos.theta = quaternion_to_yaw(msg.pose.orientation)
        self.get_logger().info(
            f"Goal set: x={self.goal_pos.x:.2f}  y={self.goal_pos.y:.2f}"
            f"  θ={self.goal_pos.theta:.2f}"
        )

        # Re-plan from current odometry position (or start_pos if not yet moving)
        plan_start = self.start_pos
        if self.current_pose is not None:
            plan_start = Pos(
                x=self.current_pose[0],
                y=self.current_pose[1],
                theta=self.current_pose[2],
            )

        self.traj = self.plan(plan_start, self.goal_pos)
        self.path_index = 0

        if self.traj:
            self.get_logger().info(f"Path planned: {len(self.traj)} waypoints.")
            self.path.poses.clear()  # type: ignore[attr-defined]
            for pos in self.traj:
                ps = PoseStamped()
                ps.header.frame_id = "odom"
                ps.pose.position.x = pos.x
                ps.pose.position.y = pos.y
                qz = np.sin(pos.theta / 2.0)
                qw = np.cos(pos.theta / 2.0)
                ps.pose.orientation.z = float(qz)
                ps.pose.orientation.w = float(qw)
                self.path.poses.append(ps)  # type: ignore[attr-defined]
            self.path_pub.publish(self.path)
            self.goal_pose = [self.traj[0].x, self.traj[0].y, self.traj[0].theta]
        else:
            self.get_logger().warn("A* failed to plan a path.")

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        if self.current_pose is None or self.goal_pose is None:
            return

        # ── Waypoint switching ────────────────────────────────────────
        dx = self.goal_pose[0] - self.current_pose[0]
        dy = self.goal_pose[1] - self.current_pose[1]
        dist = np.hypot(dx, dy)

        if dist < 0.1:
            if self.path_index >= len(self.traj) - 1:
                self.get_logger().info("Final goal reached — stopping.")
                self._stop()
                return
            self.path_index += 1
            wp = self.traj[self.path_index]
            self.goal_pose = [wp.x, wp.y, wp.theta]
            self.get_logger().info(
                f"Waypoint {self.path_index}/{len(self.traj)-1}  "
                f"hitch={np.degrees(self.hitch_angle):.1f}°"
            )

        # ── MPC solve ────────────────────────────────────────────────
        optimal_controls, predicted_traj = self.mpc.solve_control(
            self.current_pose, self.goal_pose
        )

        v, omega = optimal_controls[0]
        self.get_logger().info(
            f"pose=({self.current_pose[0]:.2f},{self.current_pose[1]:.2f},"
            f"{np.degrees(self.current_pose[2]):.1f}°)  "
            f"hitch={np.degrees(self.hitch_angle):.1f}°  "
            f"v={v:.3f}  ω={omega:.3f}"
        )

        # ── Publish command ───────────────────────────────────────────
        cmd = Twist()
        cmd.linear.x  = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

        # ── Publish MPC predicted path for RViz ──────────────────────
        self.mpc_path.poses.clear()  # type: ignore[attr-defined]
        for pos in predicted_traj:
            ps = PoseStamped()
            ps.header.frame_id = "odom"
            ps.pose.position.x = pos.x
            ps.pose.position.y = pos.y
            qz = np.sin(pos.theta / 2.0)
            qw = np.cos(pos.theta / 2.0)
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            self.mpc_path.poses.append(ps)  # type: ignore[attr-defined]
        self.mpc_path_pub.publish(self.mpc_path)

    def _stop(self):
        self.cmd_pub.publish(Twist())
        self.goal_pose = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rclpy.init()

    my_mpc = MPC(dt=0.1, horizon=20)
    my_mpc.set_physical_constraints(v_max=0.25, v_min=-0.25, omega_max=0.5)
    my_mpc.set_weights(
        position_weight=1.0,
        heading_weight=0.5,
        control_weight=0.1,
        phi_weight=3.0,      # penalise hitch angle heavily — keeps trailer straight
    )

    node = TurtleBotMPCNode(my_mpc)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()