"""Node for viewing yolo debug images and extracting bbox data from the drawn boxes."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import csv


QOS_RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

# Image dimensions from your bag
IMG_W = 1280
IMG_H = 720


class BagImageViewer(Node):
    """Subscribes to debug image topic, displays frames, and extracts bbox data."""

    def __init__(self):
        super().__init__('bag_image_viewer')

        self.bridge = CvBridge()
        self.frame_count = 0
        self.paused = False

        # CSV logging of extracted bbox data
        self.csv_path = 'extracted_detections.csv'
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'frame', 'timestamp_sec',
            'cx_px', 'cy_px',           # Raw pixel center
            'cx_norm', 'cy_norm',        # Normalized (-1 to 1) as your controller uses
            'size_x_px', 'size_y_px',   # Raw pixel size
            'size_norm',                 # Normalized area as your controller uses
            'aspect_ratio',             # size_x / size_y
            'detected'
        ])

        self.image_sub = self.create_subscription(
            Image,
            '/yolo/debug_image',
            self.image_cb,
            QOS_RELIABLE,
        )

        cv2.namedWindow('YOLO Debug Playback', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Debug Playback', 1280, 720)

        self.get_logger().info(
            f'Bag image viewer ready. Logging bbox data to {self.csv_path}\n'
            'Controls: [SPACE] pause/resume | [s] save frame | [q] quit'
        )

    def _extract_bboxes(self, frame: np.ndarray) -> list:
        """
        Extract bounding boxes from the drawn green rectangles in the image.
        Returns list of (cx, cy, w, h) in pixels.
        """
        # Isolate the green channel - YOLO draws in bright green (0, 255, 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green hue range in HSV
        lower_green = np.array([40, 150, 150])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter out tiny noise contours (text labels, etc.)
            if w > 15 and h > 15:
                cx = x + w / 2
                cy = y + h / 2
                bboxes.append((cx, cy, w, h))

        return bboxes

    def image_cb(self, msg: Image):
        """Display each incoming image frame and extract bbox data."""
        if self.paused:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        self.frame_count += 1
        timestamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Extract bboxes from green drawn boxes
        bboxes = self._extract_bboxes(frame)
        detected = len(bboxes) > 0

        for (cx_px, cy_px, w_px, h_px) in bboxes:
            # Normalize to match your controller's convention
            cx_norm = (cx_px - (IMG_W / 2.0)) / (IMG_W / 2.0)
            cy_norm = (cy_px - (IMG_H / 2.0)) / (IMG_H / 2.0)
            size_norm = (w_px / 640.0) * (h_px / 480.0)
            # aspect = w_px / h_px if h_px > 0 else 0

            self.csv_writer.writerow([
                self.frame_count, f'{timestamp_sec:.4f}',
                f'{cx_px:.1f}', f'{cy_px:.1f}',
                f'{cx_norm:.4f}', f'{cy_norm:.4f}',
                f'{w_px:.1f}', f'{h_px:.1f}',
                f'{size_norm:.5f}',
                # f'{aspect:.3f}',
                detected
            ])

            # Draw extracted data on frame for verification
            cv2.putText(frame, f'cx={cx_norm:.2f} cy={cy_norm:.2f}',
                        (int(cx_px), int(cy_px) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f'size={size_norm:.4f} ',
                        (int(cx_px), int(cy_px) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Overlay frame info
        status = 'DETECTED' if detected else 'NO DETECTION'
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(frame, f'Frame: {self.frame_count}  t={timestamp_sec:.2f}s  {status}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('YOLO Debug Playback', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info(f'Quitting. Data saved to {self.csv_path}')
            self.csv_file.close()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord(' '):
            self.paused = not self.paused
            self.get_logger().info(f'{"Paused" if self.paused else "Resumed"}')
        elif key == ord('s'):
            filename = f'frame_{self.frame_count:04d}.png'
            cv2.imwrite(filename, frame)
            self.get_logger().info(f'Saved {filename}')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


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