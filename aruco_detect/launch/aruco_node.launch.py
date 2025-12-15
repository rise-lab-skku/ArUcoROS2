from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("aruco_detect")
    transform_file_path = os.path.join(pkg_share, "marker_transforms.npz")

    return LaunchDescription(
        [
            Node(
                package="aruco_detect",
                executable="aruco_node",
                name="aruco_node",
                output="screen",
                parameters=[
                    {"aruco_type": "DICT_5X5_100"},
                    {"aruco_length": 0.07},
                    {"aruco_transforms": transform_file_path},  # ← 상대 경로 대신 share 기반
                    {"aruco_main_marker_id": 0},
                    {"camera_img_topic": "camera/color/image_raw"},
                    {"camera_info_topic": "camera/color/camera_info"},
                    {"camera_frame_id": "camera_color_optical_frame"},
                ],
            )
        ]
    )
