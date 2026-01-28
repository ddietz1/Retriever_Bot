import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

PIPELINE = (
    'udpsrc port=5602 '
    'caps="application/x-rtp,media=video,encoding-name=JPEG,payload=96,clock-rate=90000" ! '
    'rtpjpegdepay ! jpegdec ! videoconvert ! videoscale ! '
    'video/x-raw,format=BGR,width=320,height=180 ! '  # Half resolution
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

        self.pub = self.create_publisher(Image, "/bluerov/camera/image_raw", QOS_IMAGE)
        self.bridge = CvBridge()

        self.last_sample_time = time.monotonic()
        self.restart_cooldown_until = 0.0
        
        # Diagnostic counters
        self.frame_count = 0
        self.publish_count = 0

        Gst.init(None)
        self.pipeline = Gst.parse_launch(PIPELINE)
        self.appsink = self.pipeline.get_by_name("sink")
        
        if self.appsink is None:
            raise RuntimeError("appsink not found (expected: appsink name=sink ...)")

        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self.on_sample)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info(f"GStreamer set_state PLAYING -> {ret.value_nick}")

        # Watchdog timer to restart pipeline if it wedges
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
            # Copy frame data from GStreamer buffer
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3)).copy()
            
            # Update timestamp for watchdog
            self.last_sample_time = time.monotonic()
            
            # Publish immediately
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "bluerov_camera"
            self.pub.publish(msg)
            
            # Diagnostics
            self.frame_count += 1
            self.publish_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"Received and published 30 frames")
                
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def check_pipeline_health(self):
        """Watchdog to restart pipeline if it stops receiving frames"""
        now = time.monotonic()
        if now - self.last_sample_time > 3.0 and now > self.restart_cooldown_until:
            self.get_logger().warn(
                f"No frames for {now - self.last_sample_time:.2f}s, restarting pipeline"
            )
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
