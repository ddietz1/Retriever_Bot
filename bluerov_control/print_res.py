import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

QOS_BEST_EFFORT = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

class PrintRes(Node):
    def __init__(self):
        super().__init__("print_res")
        self.sub = self.create_subscription(
            CompressedImage,
            "/bluerov/camera/image_raw/compressed",
            self.cb,
            QOS_BEST_EFFORT,
        )

    def cb(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().error("cv2.imdecode failed")
            return
        h, w = img.shape[:2]
        self.get_logger().info(f"Decoded image size: {w}x{h}  format={msg.format}")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = PrintRes()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
