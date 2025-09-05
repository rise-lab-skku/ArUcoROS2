from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('aruco_detect')

    return LaunchDescription([
        Node(
            package='aruco_detect',
            executable='aruco_calibrate',
            name='aruco_calibrate',
            output='screen',
            parameters=[
                {'aruco_type': 'DICT_5X5_100'},
                {'aruco_length': 0.06},
                {'aruco_save_dir': pkg_share},  
                {'aruco_main_marker_id': 0},
                {'camera_img_topic': '/camera/color/image_raw'},
                {'camera_info_topic': '/camera/color/camera_info'},
                {'camera_frame_id': 'camera_color_optical_frame'},
                {'calibration_duration': 15.0}
            ]
        )
    ])
