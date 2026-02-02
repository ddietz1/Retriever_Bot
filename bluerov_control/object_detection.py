import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from bluerov_interfaces.msg import Object

import cv2
import numpy as np


QOS_IMAGE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class GreenRingDetector(Node):
    def __init__(self):
        super().__init__('object_detection')

        self.bridge = CvBridge()

        # Create subscribers
        # Subscribe to image
        self.sub = self.create_subscription(
            CompressedImage,
            '/bluerov/camera/image_raw/compressed',
            self.on_image,
            QOS_IMAGE
        )

        # Create publishers
        self.pub_debug = self.create_publisher(
            CompressedImage,
            '/bluerov/ring/debug_image/compressed',
            QOS_IMAGE
        )
        self.pub_ex = self.create_publisher(
            Float32,
            '/bluerov/ring/error_x',
            10
        )
        self.pub_ey = self.create_publisher(
            Float32,
            '/bluerov/ring/error_y',
            10
        )
        self.pub_area = self.create_publisher(
            Float32,
            '/bluerov/ring/area_norm',
            10
        )
        self.object_pub = self.create_publisher(
            Object,
            '/bluerov/ring/object',
            QOS_IMAGE
        )

        # HSV thresholds for green
        # self.h_low = 35
        # self.h_high = 80
        # self.s_low = 45
        # self.v_low = 35

        # Orange
        self.h_low = 7
        self.h_high = 30
        self.s_low = 75
        self.v_low = 75

        # Optional: ignore tiny detections
        self.min_area_px = 500

        # Ignore large detections as well
        self.max_area_px = 1500
        # Diagnostic counter
        self.frame_count = 0
        # Number of detected contours in a row
        self.detected_contours = 0

    def on_image(self, msg: CompressedImage):
        # Decode compressed JPEG image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('Failed to decode compressed image')
            return

        # Diagnostics
        # self.frame_count += 1
        # if self.frame_count % 30 == 0:
        #     self.get_logger().info(f"Detection received 30 frames")

        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        debug = frame.copy()

        if len(contours) == 0:
            # No detection: publish zeros and debug view
            self.pub_ex.publish(Float32(data=0.0))
            self.pub_ey.publish(Float32(data=0.0))
            self.pub_area.publish(Float32(data=0.0))

            _, debug_buffer = cv2.imencode(
                '.jpg', debug, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            dbg_msg = CompressedImage()
            dbg_msg.header = msg.header
            dbg_msg.format = "jpeg"
            dbg_msg.data = debug_buffer.tobytes()
            self.pub_debug.publish(dbg_msg)
            self.detected_contours = 0  # Reset counter

            return

        # Largest blob
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))

        if (area < self.min_area_px):
            self.pub_ex.publish(Float32(data=0.0))
            self.pub_ey.publish(Float32(data=0.0))
            self.pub_area.publish(Float32(data=0.0))

            _, debug_buffer = cv2.imencode(
                '.jpg', debug, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            dbg_msg = CompressedImage()
            dbg_msg.header = msg.header
            dbg_msg.format = 'jpeg'
            dbg_msg.data = debug_buffer.tobytes()
            self.pub_debug.publish(dbg_msg)
            self.detected_contours = 0  # Reset counter
            return
        # Check circularity (rings are circular)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            self.detected_contours = 0
            return
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Ring should be reasonably circular (0.7-1.0)
        if circularity < 0.25:
            #self.get_logger().info(f"Rejected: circularity={circularity:.2f}")
            self.detected_contours = 0
            return

        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + bw / 2.0  # Center of rect
        cy = y + bh / 2.0

        # Normalize errors to [-1, 1]
        ex = (cx - (w / 2.0)) / (w / 2.0)
        ey = (cy - (h / 2.0)) / (h / 2.0)

        # Normalize area to [0, 1] relative to image
        area_norm = min(1.0, area / float(w * h))

        # Draw overlay
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(debug, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.line(debug, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

        cv2.putText(
            debug,
            f'ex={ex:+.2f} ey={ey:+.2f} area={area_norm:.3f}',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        object_msg = Object()

        # Publish 'detected' if contours are detected for > 50 frames
        if self.detected_contours > 50:
            object_msg.detected = True
        else:
            object_msg.detected = False

        object_msg.cx = ex
        object_msg.cy = ey
        object_msg.area = area_norm

        self.pub_ex.publish(Float32(data=float(ex)))
        self.pub_ey.publish(Float32(data=float(ey)))
        self.pub_area.publish(Float32(data=float(area_norm)))
        self.object_pub.publish(object_msg)

        _, debug_buffer = cv2.imencode(
            '.jpg', debug, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        dbg_msg = CompressedImage()
        dbg_msg.header = msg.header
        dbg_msg.format = 'jpeg'
        dbg_msg.data = debug_buffer.tobytes()
        self.pub_debug.publish(dbg_msg)

        self.detected_contours += 1


def main():
    """Run the ROS2 node."""
    rclpy.init()
    node = GreenRingDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
