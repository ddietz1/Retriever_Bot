import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy, QoSProfile, QoSHistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage
import numpy as np

QOS_IMAGE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

QOS_BAG = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

class YoloDetection(Node):
    def __init__(self):
        super().__init__('yolo_detect')

        self.model = YOLO('/home/derek-dietz/Winter_Project/ros_ws/best.pt')

        self.bridge = CvBridge()

        # Create Subscribers

        self.sub = self.create_subscription(
            CompressedImage,
            '/bluerov/camera/image_raw/compressed',
            self.yolo_callback,
            QOS_IMAGE
        )

        self.image_sub = self.create_subscription(
            Image,
            '/bluerov/camera/image_raw',
            self.bag_callback,
            QOS_BAG
        )

        self.pub = self.create_publisher(
            Detection2DArray,
            '/boundingBox',
            10
        )

        self.debug_pub = self.create_publisher(
            Image, '/yolo/debug_image', 10
        )

        self.frame_count = 0

    def bag_callback(self, msg):
        '''Callback for rosbag image msgs.'''
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if cv_image is None:
            return
        self.run_inference(cv_image)

    def yolo_callback(self, msg):
        '''Callback for live ROV compressed image msgs.'''
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            return
        self.run_inference(cv_image)

    def run_inference(self, cv_image):
        '''Shared inference logic.'''
        results = self.model(cv_image, verbose=False)[0]
        detection_arr = Detection2DArray()

        # if results:
        #     print('bounding box found.')

        for box in results.boxes:
            detection = Detection2D()
            bbox = BoundingBox2D()
            x1, y1, x2, y2 = box.xyxy[0]
            bbox.center.position.x = float((x1 + x2) / 2)
            bbox.center.position.y = float((y1 + y2) / 2)
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            detection.bbox = bbox
            detection_arr.detections.append(detection)

        self.pub.publish(detection_arr)

        # Debug overlay
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f'{conf:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            w, h = x2-x1, y2-y1
            size = (w/640.0) * (h/480.0)
            cv2.putText(cv_image, f'size={size:.3f}', (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetection()
    rclpy.spin(node)

if __name__ == '__main__':
    main()