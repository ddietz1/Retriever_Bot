"""Module for controlling the ROV velocity.

Overall structure:
Subscribe to /bluerov/ring/object,
Determine ROV velocity based on position from ring
Publish Twist messages to /bluerov/cmd_vel
When close enough, call gripper close service
"""

from enum import Enum, auto
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist, TwistStamped
from mavros_msgs.msg import ManualControl, OverrideRCIn
from mavros_msgs.srv import CommandLong

from std_msgs.msg import Float32
from std_srvs.srv import Empty
from bluerov_interfaces.srv import Pitch
from bluerov_interfaces.msg import Object

QOS_IMAGE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
# markerQoS = QoSProfile(
#             depth=10,
#             durability=QoSDurabilityPolicy.VOLATILE)


class State(Enum):
    """Current state of the system.

    Used for determining if zero vel commands should be published
    """

    RING_DETECTED = auto()
    SEARCHING = auto()


class Controller(Node):
    """Controls the ROV based on sensor input."""

    def __init__(self):
        super().__init__('Controller')

        # Declare constants
        self.eps = 0.5  # Will need tuning
        self.P_yaw = 0.5  # proportional variable
        self.P_heave = 0.55
        self.P_forward = 0.2
        self.P_strafe = 0.2
        self.target_size = 0.2  # Area at which the ROV is close enough to the ring
        self.publish_commands = False  # Automatically not publishing movement cmds

        # Create subscribers
        self.object_sub = self.create_subscription(
            Object,
            '/bluerov/ring/object',
            self.controller,
            QOS_IMAGE
        )

        # Create publishers
        self.twist_pub = self.create_publisher(
            Twist,
            '/bluerov/cmd_vel',
            10
        )
        # Create services
        # Service call to start detection logic
        self.run_detection = self.create_service(
            Empty,
            'bluerov/detect',
            self.toggle_detection,
            10
        )

        # Create clients

    def toggle_detection(self, request, response):
        """Toggle detection, if OFF then no movement commands are published."""
        self.publish_commands = True

    def controller(self, msg: Object):
        """Control the ROV based on object detection."""
        # positive right in the camera frame
        center_x = msg.cx
        # positive down in the camera frame
        center_y = msg.cy
        area = msg.area
        detected = msg.detected

        # Test that we are getting data

        # Need to test experimentally how close the ring can get
        # before it can be grabbed
        # Seems like the max area it can see is roughly 0.2
        # Camera center is 18.7 cm above bottom
        # Center of the gripper is 5.5 cm above ground
        # Center of the gripper is 4.5 cm left of the camera in its POV
        tw = Twist()
        if detected:
            diff_x = center_x
            diff_y = center_y
            diff_area = max(0.0, self.target_size - area)
            if abs(diff_x) < self.eps:
                diff_x = 0.0
            if abs(diff_y) < self.eps:
                diff_y = 0.0
            if (0.0 < area < 0.1):  # Sees the ring but kinda far
                tw.angular.z = self.P_yaw * diff_x
                tw.linear.z = self.P_heave * diff_y
                tw.linear.x = self.P_forward * diff_area
            elif (0.1 < area < self.target_size):
                tw.linear.y = self.P_strafe * diff_x
                tw.linear.z = self.P_heave * diff_y
                tw.linear.x = self.P_forward * diff_area
            self.get_logger().info(f'detected! full twist: {tw}')
            self.get_logger().info(f'area detected is {area}')
            self.get_logger().info(f'x error is {diff_x}, y error is {diff_y}')
        else:
            tw.angular.z = -0.5  # rotate slowly
            tw.linear.z = 0.0  # all else zero
            tw.linear.x = 0.0
            tw.linear.y = 0.0
        self.twist_pub.publish(tw)


def main():
    """Run the ROS2 node."""
    rclpy.init()
    node = Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
