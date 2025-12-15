
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import struct
from tf2_geometry_msgs import PoseStamped
import torch
from scipy.spatial.transform import Rotation
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_geometry_msgs import TransformStamped
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
import matplotlib.pyplot as plt
import cv2
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose


rclpy.init()

class RVizVisualizer(Node):

    def __init__(self):
        super().__init__("SchlamPublisher")
        qos_profile = QoSProfile(depth=10)  # History depth of 10 messages
        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL  # Set durability
        qos_profile.reliability = ReliabilityPolicy.RELIABLE  # Often combined with reliable delivery
        qos_profile.history = HistoryPolicy.KEEP_LAST
        self.image_publisher = self.create_publisher(Image, "/camera/old_img", qos_profile)
        self.feature_image_publisher = self.create_publisher(Image, "/camera/feature_img", qos_profile)
        self.camera_publisher = self.create_publisher(CameraInfo, "/camera/camera_info", qos_profile)
        self.br = StaticTransformBroadcaster(self)
        self.tp = TransformBroadcaster(self)
        self.bridge = CvBridge()
        self.point_publisher = self.create_publisher(PointCloud2, "/points", qos_profile)
        self.camera_path_publisher = self.create_publisher(Path, "/camera_path", qos_profile)
        self.gt_path_publisher = self.create_publisher(Path, "/gt_path", qos_profile)
        self.imu_path_publisher = self.create_publisher(Path, "/imu_path", qos_profile)
        self.frame_idx = 0
        self.imu_frame_idx = 0
        self.gt_idx = 0

    def publish_imu(self, Rs, ts):
        path = self.create_path(Rs, ts)
        self.imu_path_publisher.publish(path)

    def create_transform(self, R, t, base, child):
        trans = TransformStamped()
        trans.header.frame_id = base
        trans.child_frame_id = child
        trans.transform.translation.x = float(t[0])
        trans.transform.translation.y = float(t[1])
        trans.transform.translation.z = float(t[2])
        quat = Rotation.from_matrix(R).as_quat().astype(np.float32)
        trans.transform.rotation.x = float(quat[0])
        trans.transform.rotation.y = float(quat[1])
        trans.transform.rotation.z = float(quat[2])
        trans.transform.rotation.w = float(quat[3])
        trans.header.stamp = self.get_clock().now().to_msg()
        return trans

    def publish_transform(self, R, t, base, child):
        # ROS2 Transforms transform data represented in frame A to their representation in frame B
        trans = self.create_transform(R, t, base, child)
        self.tp.sendTransform(trans)

    def publish_static_transform(self, R, t, base, child):
        # ROS2 Transforms transform data represented in frame A to their representation in frame B
        trans = self.create_transform(R, t, base, child)
        self.br.sendTransform(trans)

    def create_path(self, Rs, ts):
        path = Path()
        path.header.frame_id = "world"
        for R, t in zip(Rs, ts):
            if isinstance(R, torch.Tensor):
                R = R.float().cpu().numpy()
            if isinstance(t, torch.Tensor):
                t = t.float().cpu().numpy()
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.pose.position.x = float(t[0])
            pose.pose.position.y = float(t[1])
            pose.pose.position.z = float(t[2])
            quat = Rotation.from_matrix(R.T).as_quat().astype(np.float32)
            pose.pose.orientation.x = float(quat[0])
            pose.pose.orientation.y = float(quat[1])
            pose.pose.orientation.z = float(quat[2])
            pose.pose.orientation.w = float(quat[3])
            path.poses.append(pose)
        return path

    def publish_camera_path(self, Rs, ts):
        path = self.create_path(Rs, ts)
        self.camera_path_publisher.publish(path)

    def publish_camera(self, img, Rs, ts, K, P, points3d, features):
        self.frame_idx = 0
        R = Rs[-1]
        t = ts[-1]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().astype(np.uint8)
        if isinstance(R, torch.Tensor):
            R = R.float().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.float().cpu().numpy()
        if isinstance(points3d, torch.Tensor):
            points3d = points3d.float().cpu().numpy()
        if isinstance(features, torch.Tensor):
            features = features.float().cpu().numpy()

        transform = TransformStamped()
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'pred/camera'
        transform.transform.translation.x = float(t[0])
        transform.transform.translation.y = float(t[1])
        transform.transform.translation.z = float(t[2])
        quat = Rotation.from_matrix(R.T).as_quat().astype(np.float32)
        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])
        self.br.sendTransform(transform)

        camera_msg = CameraInfo()
        camera_msg.header.frame_id = 'pred/camera'
        camera_msg.height = int(img.shape[0])
        camera_msg.width = int(img.shape[1])
        camera_msg.k = K.cpu().numpy().flatten().astype(np.float64)
        camera_msg.p = np.eye(4).flatten().astype(np.float64)
        self.camera_publisher.publish(camera_msg)

        img_msg = self.bridge.cv2_to_imgmsg(img)
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'pred/camera'
        self.image_publisher.publish(img_msg)

        self.point_publisher.publish(self.create_pointcloud2(points3d))

        feat_img_msg = self.bridge.cv2_to_imgmsg(self.plot_features(img, features), encoding='rgb8')
        feat_img_msg.header.stamp = self.get_clock().now().to_msg()
        feat_img_msg.header.frame_id = 'pred/camera'
        self.feature_image_publisher.publish(feat_img_msg)

    def plot_features(self, image, features):
        figure = plt.figure(figsize=(24, 12))
        ax = figure.gca()
        w, h = image.shape[-1], image.shape[-2]

        ax.scatter(features[:, 0], features[:, 1], c="r")
        ax.imshow(image.astype(np.uint8), vmin=0, vmax=255)
        figure.tight_layout()
        figure.canvas.draw()

        # Convert the canvas to a raw RGB buffer
        buf = figure.canvas.tostring_argb()
        ncols, nrows = figure.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)[:, :, 1:]
        plt.close()
        return img

    def publish_gt(self, gts):
        path = self.create_path([it[:3, :3] for it in gts], [it[:3, 3] for it in gts])
        self.gt_path_publisher.publish(path)

    def create_pointcloud2(self, points, frame_id="world"):
        """
        Create a PointCloud2 message from a Nx3 NumPy array.
        """
        header = Header()
        header.stamp = rclpy.time.Time().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Flatten and pack the data
        data = []
        for p in points:
            data.append(struct.pack('fff', *p))
        data_binary = b''.join(data)

        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = points.shape[0]
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 12  # 3 * 4 bytes
        pointcloud_msg.row_step = pointcloud_msg.point_step * points.shape[0]
        pointcloud_msg.is_dense = True
        pointcloud_msg.data = data_binary

        return pointcloud_msg

def run_async():
    minimal_publisher = RVizVisualizer()
    return minimal_publisher

