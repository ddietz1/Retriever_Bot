from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')
    gcs_url = LaunchConfiguration('gcs_url')
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
        # ExecuteProcess(
        #     cmd=[venv_python, '-m', 'bluerov_control.Depth_Estimator'],
        #     output='screen'
        # ),
    ])
