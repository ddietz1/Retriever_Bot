import threading
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

SDP_PATH = "/home/derek-dietz/Winter_Project/H264 USB Camera_ USB Camera.sdp"

PIPELINE = (
  'udpsrc port=5602 '
  'caps="application/x-rtp,media=video,encoding-name=JPEG,payload=96,clock-rate=90000" ! '
  'rtpjpegdepay ! jpegdec ! videoconvert ! videoscale ! '
  'video/x-raw,format=BGR,width=640,height=360 ! '
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

        self.frame_count = 0

        self.pub = self.create_publisher(Image, "/bluerov/camera/image_raw", QOS_IMAGE)
        self.bridge = CvBridge()

        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_id = 0
        self.last_published_id = -1

        self.last_sample_time = time.monotonic()
        self.restart_cooldown_until = 0.0

        Gst.init(None)
        self.pipeline = Gst.parse_launch(PIPELINE)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.set_property("emit-signals", True)
        if self.appsink is None:
            raise RuntimeError("appsink not found (expected: appsink name=sink ...)")

        self.appsink.connect("new-sample", self.on_sample)

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info(f"GStreamer set_state PLAYING -> {ret.value_nick}")

        # Publish at up to 30Hz (only publishes when a new frame arrives)
        self.timer = self.create_timer(1.0 / 60.0, self.publish_latest)

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
            frame_view = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3))
            with self.lock:
                if self.latest_frame is None or self.latest_frame.shape != (height, width, 3):
                    self.latest_frame = np.empty((height, width, 3), dtype=np.uint8)
                np.copyto(self.latest_frame, frame_view)
                self.frame_id += 1
                self.last_sample_time = time.monotonic()
        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def publish_latest(self):
        # Gentle watchdog: restart only if truly wedged
        now = time.monotonic()
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info("Received 30 frames")
            self.frame_count = 0
        if now - self.last_sample_time > 3.0 and now > self.restart_cooldown_until:
            self.get_logger().warn(
                f"No MJPEG samples for {now - self.last_sample_time:.2f}s, restarting pipeline"
            )
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.set_state(Gst.State.READY)
            self.pipeline.set_state(Gst.State.PLAYING)
            self.restart_cooldown_until = now + 10.0
            return

        with self.lock:
            if self.latest_frame is None:
                return
            if self.frame_id == self.last_published_id:
                return
            self.last_published_id = self.frame_id
            frame = self.latest_frame.copy()

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "bluerov_camera"
        self.pub.publish(msg)
        self.get_logger().info(f"Publish took {(time.monotonic() - now)*1000:.2f}ms")


def main():
    rclpy.init()
    node = MJPEGCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
