import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np
import open3d as o3d
import struct

class StairDetectionNode(Node):
    def __init__(self):
        super().__init__('stair_detection')
        
        # Subscriber for Livox LiDAR point cloud
        self.subscription = self.create_subscription(
            PointCloud2, '/livox/lidar', self.lidar_callback, 10)

        # Publisher for detected stairs
        self.stair_publisher = self.create_publisher(String, '/stairs_detected', 10)

    def lidar_callback(self, msg):
        """Processes the incoming point cloud and detects stairs using RANSAC."""
        points = self.parse_point_cloud(msg)
        if len(points) == 0:
            return

        stair_planes = self.detect_stairs(points)
        self.publish_stairs(stair_planes)

    def parse_point_cloud(self, msg):
        """ Extracts XYZ points from PointCloud2 message. """
        points = []
        point_step = msg.point_step
        data = msg.data
        for i in range(0, len(data), point_step):
            x, y, z = struct.unpack_from('fff', data, offset=i)
            points.append([x, y, z])
        return np.array(points)

    def detect_stairs(self, points):
        """ Applies RANSAC to detect stair-like planes. """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        plane_models = []
        max_iterations = 100
        distance_threshold = 0.02  # Adjust for better plane fitting

        while len(point_cloud.points) > 1000:  # Stop if too few points left
            plane_model, inliers = point_cloud.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=max_iterations
            )
            
            # Extract stair step (if plane is horizontal)
            normal = np.abs(plane_model[:3])
            if normal[2] > 0.9:  # Stairs should be mostly horizontal
                plane_models.append(plane_model)

            # Remove inlier points (detected plane)
            inlier_cloud = point_cloud.select_by_index(inliers)
            point_cloud = point_cloud.select_by_index(inliers, invert=True)

        return plane_models

    def publish_stairs(self, stair_planes):
        """ Publishes detected stair plane parameters. """
        if stair_planes:
            msg = String()
            msg.data = f"Detected {len(stair_planes)} stair steps!"
            self.stair_publisher.publish(msg)
            self.get_logger().info(msg.data)
        else:
            self.get_logger().info("No stairs detected.")

def main(args=None):
    rclpy.init(args=args)
    node = StairDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
