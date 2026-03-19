#!/usr/bin/env python3
"""
TurtleBot MPC Node — reverse trailer parking.

Flow
────
1. User publishes a goal_pos (PoseStamped) in RViz on topic /goal_pos.
2. Node plans a REVERSE path from current odometry to the goal using
   the trailer-aware Hybrid A*.
3. MPC tracks the path waypoint-by-waypoint in reverse, using the live
   hitch angle from /joint_states at every control step.
4. Robot stops when the final waypoint is reached (trailer parked).

Topics
──────
  Sub  /odom              nav_msgs/Odometry
  Sub  /joint_states      sensor_msgs/JointState
  Sub  /goal_pos          geometry_msgs/PoseStamped     ← set in RViz
  Sub  /start_pos         geometry_msgs/PoseWithCovarianceStamped  (optional)
  Pub  /cmd_vel           geometry_msgs/Twist
  Pub  /trajectory        nav_msgs/Path   (A* path)
  Pub  /mpc_trajectory    nav_msgs/Path   (MPC predicted horizon)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
    Quaternion,
)
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from hybrid_a_star_planner.mpc import MPC
from hybrid_a_star_planner.a_star import AStar, Pos


HITCH_JOINT = "pivot_marker_drawbar_joint"

# Waypoint acceptance radius (metres) and heading tolerance (rad)
WP_DIST_TOL  = 0.20
WP_HEAD_TOL  = 0.15


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _make_pose_stamped(x: float, y: float, theta: float) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = "base_footprint"
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.orientation.z = float(np.sin(theta / 2.0))
    ps.pose.orientation.w = float(np.cos(theta / 2.0))
    return ps


class TrailerParkingNode(Node):
    def __init__(self, mpc_controller: MPC, planner: AStar):
        super().__init__("trailer_parking_node")
        self.mpc     = mpc_controller
        self.planner = planner

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Publishers ────────────────────────────────────────────────
        self.cmd_pub      = self.create_publisher(Twist,       "/cmd_vel",        10)
        self.path_pub     = self.create_publisher(Path,        "/trajectory",     10)
        self.mpc_path_pub = self.create_publisher(Path,        "/mpc_trajectory", 10)

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(Odometry,   "/odom",         self._odom_cb,   qos_be)
        self.create_subscription(JointState, "/joint_states", self._joint_cb,  10)
        self.create_subscription(PoseStamped,"/goal_pos",     self._goal_cb,   1)
        self.create_subscription(
            PoseWithCovarianceStamped, "/start_pos", self._start_cb, 1
        )

        # ── State ─────────────────────────────────────────────────────
        self.current_pose: list[float] | None = None   # [x, y, theta]
        self.hitch_angle: float                = 0.0

        self.traj:       list[Pos] = []
        self.wp_index:   int       = 0
        self.goal_wp:    list[float] | None = None     # current waypoint [x,y,θ]
        self.parking:    bool = False                  # True while following path

        # ── Control timer ─────────────────────────────────────────────
        self.timer = self.create_timer(self.mpc.dt, self._control_loop)
        self.get_logger().info("Trailer Parking Node ready.  "
                               "Publish a PoseStamped to /goal_pos to start.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_pose = [p.x, p.y, float(np.arctan2(siny, cosy))]

    def _joint_cb(self, msg: JointState):
        if HITCH_JOINT in msg.name:
            idx = msg.name.index(HITCH_JOINT)
            self.hitch_angle = float(msg.position[idx])
            self.mpc.update_hitch_angle(self.hitch_angle)

    def _start_cb(self, msg: PoseWithCovarianceStamped):
        # Manual override of start position (optional)
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        theta = quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_pose = [x, y, theta]
        self.get_logger().info(f"Start override: ({x:.2f}, {y:.2f}, {np.degrees(theta):.1f}°)")

    def _goal_cb(self, msg: PoseStamped):
        if self.current_pose is None:
            self.get_logger().warn("No odometry yet — ignoring goal.")
            return

        gx    = msg.pose.position.x
        gy    = msg.pose.position.y
        gth   = quaternion_to_yaw(msg.pose.orientation)
        self.get_logger().info(
            f"Goal received: ({gx:.2f}, {gy:.2f}, {np.degrees(gth):.1f}°)"
        )

        start = Pos(x=self.current_pose[0],
                    y=self.current_pose[1],
                    theta=self.current_pose[2])
        goal  = Pos(x=gx, y=gy, theta=gth)

        self.get_logger().info("Planning reverse path …")
        path = self.planner.plan(start, goal)

        if not path:
            self.get_logger().error("Planner returned an empty path — aborting.")
            return

        self.traj     = path
        self.wp_index = 0
        self.goal_wp  = [path[0].x, path[0].y, path[0].theta]
        self.parking  = True
        self.mpc.u_prev = None   # reset warm-start for new manoeuvre

        self.get_logger().info(f"Path planned: {len(path)} waypoints.  Starting reverse park.")

        # Publish full A* path for RViz
        ros_path = Path()
        ros_path.header.frame_id = "base_footprint"
        for wp in path:
            ros_path.poses.append(_make_pose_stamped(wp.x, wp.y, wp.theta))
        self.path_pub.publish(ros_path)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        if not self.parking or self.current_pose is None or self.goal_wp is None:
            return

        cx, cy, cth = self.current_pose
        gx, gy, gth = self.goal_wp

        dist     = np.hypot(gx - cx, gy - cy)
        head_err = abs(np.arctan2(np.sin(gth - cth), np.cos(gth - cth)))

        # ── Waypoint switch ───────────────────────────────────────────
        if dist < WP_DIST_TOL and head_err < WP_HEAD_TOL:
            if self.wp_index >= len(self.traj) - 1:
                self.get_logger().info(
                    f"Trailer parked!  Final hitch angle: "
                    f"{np.degrees(self.hitch_angle):.1f}°"
                )
                self._stop()
                return
            self.wp_index += 1
            wp = self.traj[self.wp_index]
            self.goal_wp = [wp.x, wp.y, wp.theta]
            self.get_logger().info(
                f"Waypoint {self.wp_index}/{len(self.traj)-1}  "
                f"dist={dist:.2f} m  φ={np.degrees(self.hitch_angle):.1f}°"
            )

        # ── MPC solve ────────────────────────────────────────────────
        try:
            optimal_controls, predicted_traj = self.mpc.solve_control(
                self.current_pose, self.goal_wp
            )
        except Exception as e:
            self.get_logger().error(f"MPC solve failed: {e}")
            self._stop()
            return

        v, omega = optimal_controls[0]

        # Safety: hard-clamp to reverse only
        v = float(np.clip(v, self.mpc.v_min, 0.0))

        self.get_logger().debug(
            f"pose=({cx:.2f},{cy:.2f},{np.degrees(cth):.1f}°)  "
            f"wp=({gx:.2f},{gy:.2f},{np.degrees(gth):.1f}°)  "
            f"dist={dist:.2f}  φ={np.degrees(self.hitch_angle):.1f}°  "
            f"v={v:.3f}  ω={omega:.3f}"
        )

        # ── Publish command ───────────────────────────────────────────
        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

        # ── Publish MPC horizon for RViz ──────────────────────────────
        mpc_path = Path()
        mpc_path.header.frame_id = "base_footprint"
        for pos in predicted_traj:
            mpc_path.poses.append(_make_pose_stamped(pos.x, pos.y, pos.theta))
        self.mpc_path_pub.publish(mpc_path)

    def _stop(self):
        self.cmd_pub.publish(Twist())
        self.parking = False
        self.goal_wp = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rclpy.init()

    # MPC — reverse-only, trailer-aware
    mpc = MPC(dt=0.1, horizon=15)
    mpc.set_physical_constraints(
        v_max     =  0.0,    # no forward motion during parking
        v_min     = -0.20,   # slow reverse
        omega_max =  0.45,
    )
    mpc.set_weights(
        position_weight = 1.0,
        heading_weight  = 0.8,
        control_weight  = 0.1,
        phi_weight      = 4.0,   # strong hitch straightening
    )

    # Planner
    planner = AStar()
    # Optionally add obstacles:
    # planner.set_obstacles([(cx, cy, half_x, half_y), ...])

    node = TrailerParkingNode(mpc, planner)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()