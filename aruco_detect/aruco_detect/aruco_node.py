#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, Pose
from sensor_msgs.msg import Image, CameraInfo
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R


# -----------------------------
# ArUco dictionary definition
# -----------------------------
ARUCO_DICT = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_5X5_1000": aruco.DICT_5X5_1000,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_6X6_1000": aruco.DICT_6X6_1000,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL
}


class ArucoNode(Node):
    """
    Publish per-marker TF as `camera_frame_id` -> `marker_<id>`.
    Additionally, publish Pose messages for each marker on topic `aruco_<id>`.
    """

    def __init__(self):
        super().__init__('aruco_marker_detect')

        # Parameters
        self.declare_parameter('aruco_type', 'DICT_6X6_100')
        self.declare_parameter('aruco_length', 0.0489)
        self.declare_parameter('camera_img_topic', '/camera/rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/rgb/camera_info')
        self.declare_parameter('camera_frame_id', 'rgb_camera_link')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('draw_axes_length', 0.03)

        # Get parameters
        self.marker_type = self.get_parameter('aruco_type').get_parameter_value().string_value
        self.marker_size = self.get_parameter('aruco_length').get_parameter_value().double_value
        self.camera_img_topic = self.get_parameter('camera_img_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_frame_id = self.get_parameter('camera_frame_id').get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.draw_axes_length = self.get_parameter('draw_axes_length').get_parameter_value().double_value

        # Publishers
        self.bridge = CvBridge()
        if self.publish_debug_image:
            self.aruco_pub = self.create_publisher(Image, "aruco_img", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Dynamic publishers for each marker ID
        self.pose_publishers = {}  # id -> Publisher(Pose)

        # Subscribers
        self.image_sub = self.create_subscription(Image, self.camera_img_topic, self.img_cb, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_cb, 10)

        # Camera intrinsics
        self.K = None
        self.D = None

        self.get_logger().info("ArucoNode started (TF + per-ID Pose publisher).")

    def info_cb(self, msg: CameraInfo):
        """Store camera intrinsics once."""
        self.K = np.reshape(msg.k, (3, 3))
        self.D = np.array(msg.d, dtype=np.float64)
        self.destroy_subscription(self.info_sub)
        self.info_sub = None
        self.get_logger().info("Camera intrinsics received and subscription removed.")

    def img_cb(self, msg: Image):
        if self.K is None:
            return

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        annotated, detections = self.detect_and_estimate(frame_bgr)

        now = self.get_clock().now().to_msg()
        for det in detections:
            marker_id = det["id"]
            pose = det["pose"]

            # TF
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now
            tf_msg.header.frame_id = self.camera_frame_id
            tf_msg.child_frame_id = f"marker_{marker_id}"
            tf_msg.transform.translation.x = pose.position.x
            tf_msg.transform.translation.y = pose.position.y
            tf_msg.transform.translation.z = pose.position.z
            tf_msg.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

            # Dynamic Pose publisher (topic: aruco_<id>)
            if marker_id not in self.pose_publishers:
                topic = f"aruco_{marker_id}"
                self.pose_publishers[marker_id] = self.create_publisher(Pose, topic, 10)
                self.get_logger().info(f"Created publisher for {topic}")

            self.pose_publishers[marker_id].publish(pose)

        if self.publish_debug_image:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            out_msg.header = msg.header
            self.aruco_pub.publish(out_msg)

    def _get_dict(self, code):
        if hasattr(aruco, 'getPredefinedDictionary'):
            return aruco.getPredefinedDictionary(code)
        return aruco.Dictionary_get(code)

    def _get_params_and_detector(self, aruco_dict):
        new_api = hasattr(aruco, 'ArucoDetector') and hasattr(aruco, 'DetectorParameters')
        if new_api:
            params = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, params)
            return params, detector, True
        else:
            params = aruco.DetectorParameters_create()
            return params, None, False

    def detect_and_estimate(self, img_bgr):
        aruco_dict = self._get_dict(ARUCO_DICT[self.marker_type])
        params, detector, use_new = self._get_params_and_detector(aruco_dict)

        if use_new:
            corners, ids, _ = detector.detectMarkers(img_bgr)
        else:
            corners, ids, _ = aruco.detectMarkers(img_bgr, aruco_dict, parameters=params)

        annotated = img_bgr.copy()
        detections = []

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.K, self.D)
            for i in range(len(ids)):
                marker_id = int(np.asarray(ids[i]).item())
                rvec = np.squeeze(rvecs[i])
                tvec = np.squeeze(tvecs[i])
                aruco.drawDetectedMarkers(annotated, [corners[i]], ids[i])
                cv2.drawFrameAxes(annotated, self.K, self.D, rvec, tvec, self.draw_axes_length)
                pose = self._rvec_tvec_to_pose(rvec, tvec)
                detections.append({"id": marker_id, "pose": pose})
        return annotated, detections

    def _rvec_tvec_to_pose(self, rvec, tvec) -> Pose:
        rot = R.from_rotvec(rvec)
        qx, qy, qz, qw = rot.as_quat()
        pose = Pose()
        pose.position.x = float(tvec[0])
        pose.position.y = float(tvec[1])
        pose.position.z = float(tvec[2])
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
