from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            namespace='camera',
            output='screen',
            parameters=[{
                'enable_color':     True,
                'enable_depth':     True,
                'enable_infra':     False,
                'enable_infra1':    False,
                'enable_infra2':    False,
                # Force RGB & Depth resolution
                'depth_module.profile': '848,480,15',  #1280,720,15
                'rgb_camera.profile':   '848,480,15',

                # 'align_depth': True
            }]
        )
    ])
