from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, NotSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition
import os


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')
    gcs_url = LaunchConfiguration('gcs_url')
    use_yolo = LaunchConfiguration('use_yolo')
    # ws = os.path.expanduser('~/Winter_Project/ros_ws')
    # venv_python = os.path.join(ws, '.venv', 'bin', 'python')

    return LaunchDescription([
        DeclareLaunchArgument(
            'fcu_url',
            default_value='udp://@14550',
            description='MAVLink FCU URL for MAVROS (e.g. udp://@14550)'
        ),
        DeclareLaunchArgument(
            'gcs_url',
            default_value='udp://192.168.2.2:14551',
            description='MAVLink GCS URL for MAVROS router forwarding (e.g. udp://192.168.2.2:14551)'
        ),
        DeclareLaunchArgument(
            'use_yolo',
            default_value='true',
            description='Determines if YOLO detection or object detection is used.'
        ),

        # Start MAVROS (now forwarding to gcs_url as well)
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'mavros', 'mavros_node',
                '--ros-args',
                '-p', ['fcu_url:=', fcu_url],
                '-p', ['gcs_url:=', gcs_url],
                # Optional, if you want to be explicit:
                # '-p', 'tgt_system:=1',
                # '-p', 'tgt_component:=1',
            ],
            output='screen'
        ),

        Node(
            package='bluerov_control',
            executable='bridge_node',
            name='bridge_node',
            output='screen',
        ),
        Node(
            package='bluerov_control',
            executable='object_detect',
            name='object_detection',
            output='screen',
            condition=IfCondition(NotSubstitution(use_yolo)),
        ),
        Node(
            package='bluerov_control',
            executable='camera_node',
            name='camera_node',
            output='screen',
        ),
        Node(
            package='bluerov_control',
            executable='control_node',
            name='control_node',
            output='screen',
        ),
        Node(
            package='bluerov_control',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            condition=IfCondition(use_yolo),
        ),
        Node(
            package='bluerov_heading',
            executable='heading_node',
            name='heading_node',
            output='screen',
        )
        # Node(
        #     package='magnetometer_compass',
        #     executable='magnetometer_compass_node',
        #     name='magnetometer_compass',
        #     remappings=[("/imu", "/mavros/imu/data"),("/mag", "/mavros/imu/mag"),]
        # )
        # ExecuteProcess(
        #     cmd=[venv_python, '-m', 'bluerov_control.Depth_Estimator'],
        #     output='screen'
        # ),
    ])
