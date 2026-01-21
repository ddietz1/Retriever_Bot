'''Module for creating the ros->ArduSub bridge.
'''

from enum import auto, Enum
# from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from geometry_msgs.msg import Twist, Vector3, TwistStamped

from std_msgs.msg import Empty, Float32

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from rclpy.duration import Duration
from rclpy.time import Time
from mavros_msgs.srv import CommandLong


class State(Enum):
    """Current state of the system.

    Used for determining if zero vel commands should be published
    """

    STOPPED = auto()
    MOVING = auto()


class Bridge(Node):
    """Creates the ROS->ArduSub bridge."""

    def __init__(self):
        """Initialize necessary objects."""
        super().__init__('bridge_node')

        # Declare constants
        self.brightness = 0  # default off
        self.lights_RC_Channel = 9
        self.light_servo = 9
        self.light_hz = 10.0
        # Declare params
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('cmd_timeout_s', 0.4)

        self.declare_parameter('max_surge', 0.3)   # m/s
        self.declare_parameter('max_sway', 0.3)
        self.declare_parameter('max_heave', 0.3)
        self.declare_parameter('max_yaw_rate', 0.5)  # rad/s

        self.declare_parameter('lights_pwm_min', 1100)
        self.declare_parameter('lights_pwm_max', 1900)
        self.declare_parameter('lights_servo', 11)

        # Get params
        self.get_parameter('publish_rate_hz').value
        self.cmd_timeout = Duration(
            seconds=self.get_parameter('cmd_timeout_s').value
        )

        self.max_surge = self.get_parameter('max_surge').value
        self.max_sway = self.get_parameter('max_sway').value
        self.max_heave = self.get_parameter('max_heave').value
        self.max_yaw_rate = self.get_parameter('max_yaw_rate').value

        self.last_cmd = Twist()
        self.last_cmd_time = None
        self.lights_dirty = True
        self.last_sent_pwm = None

        markerQoS = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE)

        # Create publishers
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            markerQoS
        )

        # self.light_brightness = self.create_publisher(
        #     OverrideRCIn,
        #     '/mavros/rc/override',
        #     markerQoS
        # )

        # Create subscriptions
        self.vel_sub = self.create_subscription(
            Twist,
            '/bluerov/cmd_vel',
            self.rov_twist_sub,
            markerQoS
        )

        self.lights_sub = self.create_subscription(
            Float32,
            '/bluerov/brightness',
            self.lights_callback,
            markerQoS
        )

        # Create services
        # Service for setting servo to toggle lights
        # self.toggle_lights_srv = self.create_service(
        #     CommandLong,
        #     'toggleLights',
        #     self.toggle_lights
        # )

        # Create Clients
        self.cmd_cli = self.create_client(CommandLong, '/mavros/cmd/command')
        # Create Timers
        self.vel_timer = self.create_timer(
            1/50,
            self.publish_vel_commands
        )

        self.light_timer = self.create_timer(
            1/10,
            self.send_lights_command
        )

    def limit_vel(self, val, limit):
        return max(-limit, min(limit, val))

    # def toggle_lights(self, request, response):
    #     """Turn lights on/off."""
    #     self.brightness = request.data
    #     pwm = 1100 + int(self.brightness) * (1900 - 1100)
    #     cmd = CommandLong()
        
    def rov_twist_sub(self, msg: Twist):
        """Subscribe to twist messages from user."""
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def lights_callback(self, msg: Float32):
        b = float(msg.data)
        self.brightness = max(0.0, min(1.0, b))
        self.lights_dirty = True
    
    def send_lights_command(self):
        if not self.lights_dirty:
            return

        # wait for MAVROS service to exist
        if not self.cmd_cli.service_is_ready():
            # don’t spam logs; log occasionally if you want
            return

        pwm = self.brightness_to_pwm()

        # Optional: only send if changed
        if self.last_sent_pwm is not None and pwm == self.last_sent_pwm:
            self.lights_dirty = False
            return

        servo_out = int(self.get_parameter('lights_servo').value)

        req = CommandLong.Request()
        req.command = 183  # MAV_CMD_DO_SET_SERVO
        req.confirmation = 0
        req.param1 = float(servo_out)  # servo output number
        req.param2 = float(pwm)        # PWM
        req.param3 = 0.0
        req.param4 = 0.0
        req.param5 = 0.0
        req.param6 = 0.0
        req.param7 = 0.0

        self.cmd_cli.call_async(req)

        self.last_sent_pwm = pwm
        self.lights_dirty = False

    def brightness_to_pwm(self) -> int:
        pwm_min = int(self.get_parameter('lights_pwm_min').value)
        pwm_max = int(self.get_parameter('lights_pwm_max').value)
        pwm = pwm_min + int(round(self.brightness * (pwm_max - pwm_min)))
        return max(pwm_min, min(pwm_max, pwm))

    # def publish_brightness(self):
    #     pwm = 1100 + int(self.brightness) * (1900 - 1100)
    #     msg = OverrideRCIn()
    #     msg.channels = [0] * 18
    #     msg.channels[self.lights_RC_Channel] = pwm
    #     self.light_brightness.publish(msg)

    def publish_vel_commands(self):
        """Timer callback."""
        now = self.get_clock().now()

        # Publish zero vel if last command has timed out
        if self.last_cmd_time is None or \
                (now - self.last_cmd_time) > self.cmd_timeout:
            cmd = Twist()  # Create a zero Twist to publish
        else:
            cmd = self.last_cmd

        cmd.linear.x = self.limit_vel(cmd.linear.x, self.max_surge)
        cmd.linear.y = self.limit_vel(cmd.linear.y, self.max_sway)
        cmd.linear.z = self.limit_vel(cmd.linear.z, self.max_heave)
        cmd.angular.z = self.limit_vel(cmd.angular.z, self.max_yaw_rate)

        # Publish commands
        out = TwistStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = 'base_link'
        out.twist = cmd

        self.vel_pub.publish(out)


def main():
    rclpy.init()
    node = Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
        