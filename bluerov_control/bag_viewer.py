"""Node for viewing yolo debug images from a ROS2 bag file."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


QOS_RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class BagImageViewer(Node):
    """Subscribes to debug image topic and displays frames with OpenCV."""

    def __init__(self):
        super().__init__('bag_image_viewer')

        self.bridge = CvBridge()
        self.frame_count = 0
        self.paused = False

        self.image_sub = self.create_subscription(
            Image,
            '/yolo/debug_image',
            self.image_cb,
            QOS_RELIABLE,
        )

        cv2.namedWindow('YOLO Debug Playback', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Debug Playback', 1280, 720)

        self.get_logger().info(
            'Bag image viewer ready.\n'
            'Controls: [SPACE] pause/resume | [s] save frame | [q] quit'
        )

    def image_cb(self, msg: Image):
        """Display each incoming image frame."""
        if self.paused:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        self.frame_count += 1

        # Overlay frame count and timestamp
        timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv2.putText(
            frame,
            f'Frame: {self.frame_count}  t={timestamp_sec:.2f}s',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow('YOLO Debug Playback', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quitting viewer.')
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord(' '):
            self.paused = not self.paused
            self.get_logger().info(f'{"Paused" if self.paused else "Resumed"}')
        elif key == ord('s'):
            filename = f'frame_{self.frame_count:04d}.png'
            cv2.imwrite(filename, frame)
            self.get_logger().info(f'Saved {filename}')


def main():
    """Run the viewer node."""
    rclpy.init()
    node = BagImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()