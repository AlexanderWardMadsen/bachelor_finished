import launch
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='social_force_predictor',
            executable='sfp_ekf',
            name='sfp_ekf',
            output='screen'
        ),
        Node(
            package='social_force_layer',
            executable='social_force_predictor',
            name='social_force_predictor',
            output='screen'
        ),
        Node(
            package='costmap_2d',
            executable='costmap_2d',
            name='costmap_2d',
            output='screen'
        ),
    ])