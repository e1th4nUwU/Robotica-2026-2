#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    description_path = get_package_share_directory("robot_description")

    urdf_path = description_path + "/urdf/robot_rrr_proyecto.urdf"
    rviz_path = description_path + "/rviz/rviz_proyecto.rviz"

    urdf_xacro = Command(["xacro ", urdf_path])
    urdf_param = {"robot_description":
                  ParameterValue(urdf_xacro, value_type=str)}

    # Visualizacion en RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_path]
    )

    # Publica la descripcion URDF del robot
    robot_description_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[urdf_param]
    )

    # Nodo de cinematica: calcula trayectorias y recibe clicks de RViz
    kinematics_node = Node(
        package="robot_kinematics",
        executable="trajectory_publisher"
    )

    # Nodo de hardware: envia y recibe posiciones de las juntas
    hardware_node = Node(
        package="robot_hardware",
        executable="robot_hardware"
    )

    return LaunchDescription([
        rviz_node,
        robot_description_node,
        kinematics_node,
        hardware_node
    ])
