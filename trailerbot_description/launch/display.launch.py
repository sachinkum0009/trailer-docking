import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    urdf_file_name = "trailerbot.urdf"
    urdf_file_path = os.path.join(
        get_package_share_directory("trailerbot_description"), "urdf", urdf_file_name
    )

    with open(urdf_file_path, "r", encoding="utf-8") as infp:
        robot_description = infp.read()

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        # arguments=["-d", os.path.join(get_package_share_directory("trailerbot_description"), "rviz", "trailerbot.rviz")],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    ld.add_action(rviz_node)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_node)
    return ld
