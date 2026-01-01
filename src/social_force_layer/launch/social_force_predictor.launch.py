from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='social_force_layer',
            executable='social_force_predictor',
            name='social_force_predictor',
            output='screen',
        ),
    ])
