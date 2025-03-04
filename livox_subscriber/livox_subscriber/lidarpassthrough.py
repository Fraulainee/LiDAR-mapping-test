import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
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
        
        self.publisher = self.create_publisher(PointCloud2, '/filtered_cloud', 10)

    def lidar_callback(self, msg):
        points = self.parse_point_cloud(msg)
        # filtered_points = self.passthrough_filter(points, axis='z', min_val=0.0, max_val=2.0)
        filtered_points = self.angle_filter(points, min_angle=-np.pi/2, max_angle=np.pi/2)

        # self.get_logger().info(f'Total Points: {len(points)} | Filtered Points: {len(filtered_points)}')

        filtered_msg = self.create_pointcloud2(msg.header, filtered_points)
        self.publisher.publish(filtered_msg)
        # self.get_logger().info(f'Filtered PointCloud Published to /filtered_cloud')

    def passthrough_filter(self, points, axis='z', min_val=0.0, max_val=2.0):
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        return [p for p in points if min_val <= p[axis_idx] <= max_val]

    def angle_filter(self, points, min_angle=-np.pi/2, max_angle=np.pi/2):
        
        filtered = []
        for x, y, z in points:
            angle = np.arctan2(y, x)
            if min_angle <= angle <= max_angle:
                filtered.append((x, y, z))
        return filtered

    def parse_point_cloud(self, msg):
        
        points = []
        point_step = msg.point_step
        data = msg.data
        for i in range(0, len(data), point_step):
            x, y, z = struct.unpack_from('fff', data, offset=i)
            points.append((x, y, z))
        return points

    def create_pointcloud2(self, header, points):
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = []
        for p in points:
            cloud_data += struct.pack('fff', *p)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = cloud_data
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = LivoxSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
