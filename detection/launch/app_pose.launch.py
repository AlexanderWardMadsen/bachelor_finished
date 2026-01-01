from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='detection',
            executable='pose_app_node_v_0_2', # or 'pose_app_node' for version 0.3 video output
            name='human_tf_node',
            output='screen',
            parameters=[
                {'model_path': '/home/alex/bachelor/workspace_bach/src/detection/detection/pose_landmarker_lite.task'},
                # {'model_path': '/home/alex/bachelor/workspace_bach/src/detection/detection/pose_landmarker_full.task'},
                # {'model_path': '/home/alex/bachelor/workspace_bach/src/detection/detection/pose_landmarker_heavy.task'},
                {'max_people': 4},
            ]
        )
    ])
