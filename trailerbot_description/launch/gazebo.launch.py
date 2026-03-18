#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    z_pose = LaunchConfiguration("z_pose")
    world_name = "empty_world"
    model_name = "trailerbot"

    urdf_file_name = "trailerbot.urdf"
    urdf_file_path = os.path.join(
        get_package_share_directory("trailerbot_description"), "urdf", urdf_file_name
    )

    with open(urdf_file_path, "r", encoding="utf-8") as infp:
        robot_description = infp.read()


    ros_gz_sim = get_package_share_directory("ros_gz_sim")

    world = os.path.join(
        get_package_share_directory('trailerbot_description'),
        'worlds',
        f"{world_name}.world"
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": ["-r -s -v4 ", world], "on_exit_shutdown": "true"}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "-g -v2 "}.items(),
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    spawn_entity_node = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name",
            model_name,
            "-topic",
            "/robot_description",
            "-x",
            x_pose,
            "-y",
            y_pose,
            "-z",
            z_pose,
        ],
    )

    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            f"/world/{world_name}/model/{model_name}/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
        ],
        remappings=[
            (f"/world/{world_name}/model/{model_name}/joint_state", "/joint_states"),
        ],
    )

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument("x_pose", default_value="0.0"))
    ld.add_action(DeclareLaunchArgument("y_pose", default_value="0.0"))
    ld.add_action(DeclareLaunchArgument("z_pose", default_value="0.2"))

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(spawn_entity_node)
    ld.add_action(bridge_node)

    return ld