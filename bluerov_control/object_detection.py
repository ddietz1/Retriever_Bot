import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
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

        self.sub = self.create_subscription(
            CompressedImage,
            '/bluerov/camera/image_raw/compressed',
            self.on_image,
            QOS_IMAGE
        )

        self.pub_debug = self.create_publisher(
            CompressedImage,
            '/bluerov/ring/debug_image/compressed',
            QOS_IMAGE
        )
        self.pub_ex = self.create_publisher(Float32, '/bluerov/ring/error_x', 10)
        self.pub_ey = self.create_publisher(Float32, '/bluerov/ring/error_y', 10)
        self.pub_area = self.create_publisher(Float32, '/bluerov/ring/area_norm', 10)
        self.object_pub = self.create_publisher(Object, '/bluerov/ring/object', QOS_IMAGE)

        # Orange HSV thresholds
        self.h_low = 7
        self.h_high = 30
        self.s_low = 75
        self.v_low = 75

        # Reject tiny detections (in pixels)
        self.min_hull_area_px = 100.0

        # Detection debounce
        self.detected_contours = 0
        self.detected_required = 25

        # --- NEW: filtered size metric (EMA) ---
        self.size_f = 0.0
        self.ema_alpha = 0.25  # 0.2–0.35 is a good start

    def on_image(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("Failed to decode compressed image")
            return

        h, w = frame.shape[:2]
        debug = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Mask cleanup (slightly better for round targets than a square kernel)
        k = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Optional: helps when close-up turns into a thin rim
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = False
        ex = ey = 0.0
        size_norm = 0.0
        circularity = 0.0

        if contours:
            c = max(contours, key=cv2.contourArea)

            # --- Stable geometry for a ring ---
            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull))
            hull_perim = float(cv2.arcLength(hull, True))

            if hull_area >= self.min_hull_area_px and hull_perim > 0.0:
                circularity = 4.0 * np.pi * hull_area / (hull_perim * hull_perim)

                # Very loose gate: just reject super spiky junk
                if circularity >= 0.02:
                    x, y, bw, bh = cv2.boundingRect(c)
                    cx = x + bw / 2.0
                    cy = y + bh / 2.0

                    ex = (cx - (w / 2.0)) / (w / 2.0)
                    ey = (cy - (h / 2.0)) / (h / 2.0)

                    # --- NEW: size from enclosing circle radius ---
                    (_, _), r = cv2.minEnclosingCircle(c)
                    r = float(r)

                    # Normalize radius to [0, 1] using half of the smaller image dimension as "1.0"
                    r_max = 0.5 * float(min(w, h))
                    size_norm = max(0.0, min(1.0, r / r_max))

                    valid = True

                    # Debug overlay
                    cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.circle(debug, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    cv2.circle(debug, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                    cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
                    cv2.line(debug, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

        # --- Update filter & publish ---
        if valid:
            # EMA filter for close-range stability
            self.size_f = (1.0 - self.ema_alpha) * self.size_f + self.ema_alpha * size_norm
            self.detected_contours += 1
        else:
            self.detected_contours = 0
            ex = ey = 0.0
            # Let the filter decay gently instead of snapping (optional)
            self.size_f *= 0.9

        self.pub_ex.publish(Float32(data=float(ex)))
        self.pub_ey.publish(Float32(data=float(ey)))
        self.pub_area.publish(Float32(data=float(self.size_f)))

        object_msg = Object()
        object_msg.detected = (self.detected_contours >= self.detected_required)
        object_msg.cx = float(ex)
        object_msg.cy = float(ey)
        object_msg.area = float(self.size_f)          # <-- now "stable size", not raw area
        object_msg.circularity = float(circularity)
        self.object_pub.publish(object_msg)

        ok, debug_buffer = cv2.imencode(".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            dbg_msg = CompressedImage()
            dbg_msg.header = msg.header
            dbg_msg.format = "jpeg"
            dbg_msg.data = debug_buffer.tobytes()
            self.pub_debug.publish(dbg_msg)


def main():
    rclpy.init()
    node = GreenRingDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
