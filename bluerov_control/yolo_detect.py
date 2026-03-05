import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloDetection(Node):
    def __init__(self):
        super.__init__('yolo_detect')

        self.model = YOLO('/home/derek-dietz/Winter_Project/ros_ws/yolov8n.pt')

        self.bridge = CvBridge()

        # Create Subscribers

        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            10
        )

        self.pub = self.create_publisher(
            Detection2DArray,
            '/BoundingBox',
            10
        )

    def yolo_callback(self, msg):
        '''Converts image to OpenCV.'''
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run inference
        results = self.model(cv_image)[0]

        detection_arr = Detection2DArray()

        for box in results.boxes:
            detection = Detection2D()
            bbox = BoundingBox2D()

            x1, y1, x2, y2 = box.xyxy[0]

            bbox.center.position.x = float((x1 + x2) / 2)
            bbox.center.position.y = float((y1 + y2) / 2)
            bbox.size_x = float((x2 - x1))
            bbox.size_y = float((y2 - y1))

            detection.bbox = bbox
            detection_arr.detections.append(detection)

        self.pub.publish(detection_arr)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetection()
    rclpy.spin(node)

if __name__ == '__main__':
    main()