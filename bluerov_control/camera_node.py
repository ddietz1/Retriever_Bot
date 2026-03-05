import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import numpy as np
import cv2

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

PIPELINE = (
    'udpsrc port=5602 '
    'caps="application/x-rtp,media=video,encoding-name=JPEG,payload=96,clock-rate=90000" ! '
    'rtpjpegdepay ! jpegdec ! videoconvert ! videoscale ! '
    'video/x-raw,format=BGR,width=1280,height=720 ! '
    'appsink name=sink emit-signals=true drop=true max-buffers=2 sync=false'
)

QOS_IMAGE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class MJPEGCameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # Publishers
        self.pub_compressed = self.create_publisher(
            CompressedImage, 
            "/bluerov/camera/image_raw/compressed", 
            QOS_IMAGE
        )
        self.pub_raw = self.create_publisher(
            Image, 
            "/bluerov/camera/image_raw", 
            QOS_IMAGE
        )

        self.bridge = CvBridge()
        self.last_sample_time = time.monotonic()
        self.restart_cooldown_until = 0.0
        
        # JPEG compression quality
        self.jpeg_quality = 95

        Gst.init(None)
        self.pipeline = Gst.parse_launch(PIPELINE)
        self.appsink = self.pipeline.get_by_name("sink")
        
        if self.appsink is None:
            raise RuntimeError("appsink not found (expected: appsink name=sink ...)")

        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self.on_sample)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info(f"GStreamer set_state PLAYING -> {ret.value_nick}")

        # Watchdog timer
        self.timer = self.create_timer(1.0, self.check_pipeline_health)

    def on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        caps = sample.get_caps()
        s = caps.get_structure(0)
        width = s.get_value("width")
        height = s.get_value("height")

        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        try:
            # Copy frame
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3)).copy()
            
            self.last_sample_time = time.monotonic()
            
            # --- Publish compressed ---
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            msg_compressed = CompressedImage()
            msg_compressed.header.stamp = self.get_clock().now().to_msg()
            msg_compressed.header.frame_id = "bluerov_camera"
            msg_compressed.format = "jpeg"
            msg_compressed.data = buffer.tobytes()
            self.pub_compressed.publish(msg_compressed)

            # --- Publish raw ---
            msg_raw = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg_raw.header.stamp = self.get_clock().now().to_msg()
            msg_raw.header.frame_id = "bluerov_camera"
            self.pub_raw.publish(msg_raw)
                
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def check_pipeline_health(self):
        """Restart pipeline if no frames received"""
        now = time.monotonic()
        if now - self.last_sample_time > 3.0 and now > self.restart_cooldown_until:
            self.get_logger().warn(f"No frames for {now - self.last_sample_time:.2f}s, restarting pipeline")
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.set_state(Gst.State.READY)
            self.pipeline.set_state(Gst.State.PLAYING)
            self.restart_cooldown_until = now + 10.0


def main():
    rclpy.init()
    node = MJPEGCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()