
"""Monocular depth estimation node for BlueROV2.

Subscribes to compressed camera images and publishes:
- Depth maps (as images for visualization)
- Estimated distance to detected objects (as Float32)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from bluerov_interfaces.msg import Object

import cv2
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image


QOS_IMAGE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class DepthEstimatorNode(Node):
    def __init__(self):
        super().__init__('depth_estimator')

        # Declare parameters
        self.declare_parameter('model_name', 'Intel/dpt-large')
        self.declare_parameter('process_every_n_frames', 3)  # Skip frames for performance
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('use_cuda', False)  # Set True if you have GPU
        
        # Get parameters
        model_name = self.get_parameter('model_name').value
        self.process_every_n = self.get_parameter('process_every_n_frames').value
        self.publish_debug = self.get_parameter('publish_debug_image').value
        use_cuda = self.get_parameter('use_cuda').value
        
        # Initialize model
        self.get_logger().info(f'Loading depth estimation model: {model_name}')
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.get_logger().info('Model loaded successfully')
        
        # Frame counter for skipping
        self.frame_count = 0
        
        # Object detection state (to compute distance to detected object)
        self.object_cx = None
        self.object_cy = None
        self.object_detected = False
        
        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/bluerov/camera/image_raw/compressed',
            self.on_image,
            QOS_IMAGE
        )
        
        self.object_sub = self.create_subscription(
            Object,
            '/bluerov/ring/object',
            self.on_object,
            QOS_IMAGE
        )
        
        # Publishers
        self.depth_image_pub = self.create_publisher(
            CompressedImage,
            '/bluerov/depth/image/compressed',
            QOS_IMAGE
        )
        
        self.object_distance_pub = self.create_publisher(
            Float32,
            '/bluerov/ring/estimated_distance',
            10
        )
        
        # Statistics for debugging
        self.inference_times = []
        self.stats_timer = self.create_timer(5.0, self.print_stats)

    def on_object(self, msg: Object):
        """Track the detected object position."""
        self.object_detected = msg.detected
        if msg.detected:
            self.object_cx = msg.cx  # Normalized [-1, 1]
            self.object_cy = msg.cy  # Normalized [-1, 1]
        else:
            self.object_cx = None
            self.object_cy = None

    def on_image(self, msg: CompressedImage):
        """Process compressed image and estimate depth."""
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return
        
        # Decode image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("Failed to decode compressed image")
            return
        
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Run inference
        start_time = self.get_clock().now()
        
        with torch.no_grad():
            # Prepare image
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict depth
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            
            # Convert to numpy
            depth_map = prediction.squeeze().cpu().numpy()
        
        inference_time = (self.get_clock().now() - start_time).nanoseconds / 1e6
        self.inference_times.append(inference_time)
        
        # Normalize depth map for visualization (inverse for better viz)
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_normalized = 1.0 - depth_normalized  # Invert: closer = brighter
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8),
            cv2.COLORMAP_MAGMA
        )
        
        # If object is detected, estimate distance to it
        if self.object_detected and self.object_cx is not None:
            # Convert normalized coords to pixel coords
            px = int((self.object_cx + 1.0) * w / 2.0)
            py = int((self.object_cy + 1.0) * h / 2.0)
            
            # Clamp to image bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            
            # Get depth value at object center (sample small region for robustness)
            region_size = 20
            x1 = max(0, px - region_size // 2)
            x2 = min(w, px + region_size // 2)
            y1 = max(0, py - region_size // 2)
            y2 = min(h, py + region_size // 2)
            
            object_depth_region = depth_map[y1:y2, x1:x2]
            object_depth = np.median(object_depth_region)  # Use median for robustness
            
            # Publish estimated distance
            distance_msg = Float32()
            distance_msg.data = float(object_depth)
            self.object_distance_pub.publish(distance_msg)
            
            # Draw on debug image
            cv2.circle(depth_colored, (px, py), 10, (0, 255, 0), 2)
            cv2.putText(
                depth_colored,
                f"Depth: {object_depth:.2f}",
                (px + 15, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Publish debug visualization
        if self.publish_debug:
            ok, buffer = cv2.imencode('.jpg', depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                depth_msg = CompressedImage()
                depth_msg.header = msg.header
                depth_msg.format = "jpeg"
                depth_msg.data = buffer.tobytes()
                self.depth_image_pub.publish(depth_msg)

    def print_stats(self):
        """Print performance statistics."""
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'Depth estimation: {avg_time:.1f}ms avg, {fps:.1f} FPS '
                f'(processing every {self.process_every_n} frames)'
            )
            self.inference_times = []


def main():
    rclpy.init()
    node = DepthEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
