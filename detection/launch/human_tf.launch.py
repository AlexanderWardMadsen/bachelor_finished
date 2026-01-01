from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='detection',
            executable='human_node',
            name='human_tf_node',
            output='screen'
        )
    ])
