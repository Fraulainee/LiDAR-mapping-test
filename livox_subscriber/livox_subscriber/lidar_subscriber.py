import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct
import numpy as np

class LivoxSubscriber(Node):
    def __init__(self):
        super().__init__('livox_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_callback,
            10)
        self.subscription  # Prevent unused variable warning

    def lidar_callback(self, msg):
        points = self.parse_point_cloud(msg)
        for x, y, z in points:
            distance = np.sqrt(x**2 + y**2 + z**2)
            self.get_logger().info(f'Point: x={x:.2f}, y={y:.2f}, z={z:.2f}, Distance={distance:.2f}m')

    def parse_point_cloud(self, msg):
        """ Extracts XYZ coordinates from PointCloud2 message. """
        points = []
        point_step = msg.point_step
        data = msg.data
        for i in range(0, len(data), point_step):
            x, y, z = struct.unpack_from('fff', data, offset=i)
            points.append((x, y, z))
        return points

def main(args=None):
    rclpy.init(args=args)
    node = LivoxSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
