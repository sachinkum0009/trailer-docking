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
        'empty_world.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": ["-r -s -v2 "], "on_exit_shutdown": "true"}.items(),
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
            "trailerbot",
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

    # robot_state_publisher_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
    #     ),
    #     launch_arguments={'use_sim_time': use_sim_time}.items()
    # )

    # spawn_turtlebot_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
    #     ),
    #     launch_arguments={
    #         'x_pose': x_pose,
    #         'y_pose': y_pose
    #     }.items()
    # )

    # set_env_vars_resources = AppendEnvironmentVariable(
    #         'GZ_SIM_RESOURCE_PATH',
    #         os.path.join(
    #             get_package_share_directory('turtlebot3_gazebo'),
    #             'models'))

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument("x_pose", default_value="0.0"))
    ld.add_action(DeclareLaunchArgument("y_pose", default_value="0.0"))
    ld.add_action(DeclareLaunchArgument("z_pose", default_value="0.2"))

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    # ld.add_action(robot_state_publisher_node)
    # ld.add_action(spawn_entity_node)
    # ld.add_action(spawn_turtlebot_cmd)
    # ld.add_action(robot_state_publisher_cmd)
    # ld.add_action(set_env_vars_resources)

    return ld