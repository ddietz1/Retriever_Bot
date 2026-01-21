from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')

    return LaunchDescription([
        DeclareLaunchArgument(
            'fcu_url',
            default_value='udp://:14540@',
            description='MAVLink FCU URL for MAVROS (e.g. udp://:14540@ or udp://@192.168.2.2:14550)'
        ),

        # Start MAVROS the same way your working manual command does.
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'mavros', 'mavros_node',
                '--ros-args',
                '-p', ['fcu_url:=', fcu_url],
                # If you want these too, uncomment:
                # '-p', 'gcs_url:=',
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
    ])
