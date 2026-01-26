import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

import cv2
import numpy as np


QOS_IMAGE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class GreenRingDetector(Node):
    def __init__(self):
        super().__init__("green_ring_detector")

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, "/bluerov/camera/image_raw", self.on_image, QOS_IMAGE
        )

        self.pub_debug = self.create_publisher(Image, "/bluerov/ring/debug_image", QOS_IMAGE)
        self.pub_ex = self.create_publisher(Float32, "/bluerov/ring/error_x", 10)
        self.pub_ey = self.create_publisher(Float32, "/bluerov/ring/error_y", 10)
        self.pub_area = self.create_publisher(Float32, "/bluerov/ring/area_norm", 10)

        # Tunable HSV thresholds for a green pool ring
        # Start here; you’ll likely adjust after seeing the debug overlay.
        self.h_low = 35
        self.h_high = 85
        self.s_low = 60
        self.v_low = 40

        # Optional: ignore tiny detections
        self.min_area_px = 500

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        debug = frame.copy()

        if len(contours) == 0:
            # No detection: publish zeros and debug view
            self.pub_ex.publish(Float32(data=0.0))
            self.pub_ey.publish(Float32(data=0.0))
            self.pub_area.publish(Float32(data=0.0))

            dbg_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            dbg_msg.header = msg.header
            self.pub_debug.publish(dbg_msg)
            return

        # Largest blob
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))

        if area < self.min_area_px:
            self.pub_ex.publish(Float32(data=0.0))
            self.pub_ey.publish(Float32(data=0.0))
            self.pub_area.publish(Float32(data=0.0))

            dbg_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            dbg_msg.header = msg.header
            self.pub_debug.publish(dbg_msg)
            return

        x, y, bw, bh = cv2.boundingRect(c)
        cx = x + bw / 2.0
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
            f"ex={ex:+.2f} ey={ey:+.2f} area={area_norm:.3f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Publish
        self.pub_ex.publish(Float32(data=float(ex)))
        self.pub_ey.publish(Float32(data=float(ey)))
        self.pub_area.publish(Float32(data=float(area_norm)))

        dbg_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        dbg_msg.header = msg.header
        self.pub_debug.publish(dbg_msg)


def main():
    rclpy.init()
    node = GreenRingDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
