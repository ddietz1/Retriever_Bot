'''Module for creating the ros->ArduSub bridge.
'''

from enum import Enum, auto
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from geometry_msgs.msg import Twist, TwistStamped
from mavros_msgs.msg import ManualControl, OverrideRCIn
from mavros_msgs.srv import CommandLong

from std_msgs.msg import Float32
from std_srvs.srv import Empty


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
        self.brightness = 0.0  # default off
        self.pitch_angle = 0.0
        self.gripper_open = False  # default closed
        self.adjusting_grip = False  # Needed for gripper servo callback
        self.adjusting_pitch = False  # Needed for pitch servo callback
        self.servo_cmd = False

        # Declare params
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('cmd_timeout_s', 0.4)

        self.declare_parameter('max_surge', 0.3)   # m/s
        self.declare_parameter('max_sway', 0.3)
        self.declare_parameter('max_heave', 0.3)
        self.declare_parameter('max_yaw_rate', 0.5)  # rad/s

        self.declare_parameter('servo_pwm_min', 1100)
        self.declare_parameter('servo_pwm_max', 1900)

        # Servo 11 function set to Gripper via BlueOS
        self.declare_parameter('gripper_servo', 11)
        # Trying to pitch the camera mount
        self.declare_parameter('mount_servo', 14)
        self.declare_parameter('z_neutral', 500)

        # Get params
        rate = self.get_parameter('publish_rate_hz').value
        self.cmd_timeout = Duration(
            seconds=self.get_parameter('cmd_timeout_s').value
        )
        self.max_surge = self.get_parameter('max_surge').value
        self.max_sway = self.get_parameter('max_sway').value
        self.max_heave = self.get_parameter('max_heave').value
        self.max_yaw = self.get_parameter('max_yaw_rate').value
        self.z_neutral = int(self.get_parameter('z_neutral').value)

        self.last_cmd = Twist()
        self.last_cmd_time = None
        self.lights_dirty = True
        self.last_sent_pwm = None

        self.cal_active = False
        self.cal_step = 0
        self.cal_step_start = None
        self.cal_cmd = Twist()

        markerQoS = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE)

        # Create publishers
        self.manual_pub = self.create_publisher(
            ManualControl,
            '/mavros/manual_control/send',
            markerQoS
        )

        self.lights_pub = self.create_publisher(
            OverrideRCIn,
            '/mavros/rc/override',
            markerQoS
        )
        # Create subscriptions
        self.vel_sub = self.create_subscription(
            Twist,
            '/bluerov/cmd_vel',
            self.rov_twist_sub,
            markerQoS
        )

        # Create Clients
        self.cmd_cli = self.create_client(CommandLong, '/mavros/cmd/command')

        # Create Services
        self.open_grip = self.create_service(
            Empty,
            '/bluerov/open_gripper',
            self.open_gripper
        )

        self.close_grip = self.create_service(
            Empty,
            '/bluerov/close_gripper',
            self.close_gripper
        )

        self.toggle_lights = self.create_service(
            Empty,
            '/bluerov/set_lights',
            self.lights_callback
        )

        self.calibrate = self.create_service(
            Empty,
            '/bluerov/calibrate',
            self.calibrate_bot
        )

        # Testing services
        # self.test_pitch = self.create_service(
        #     Empty,
        #     'test_pitch',
        #     self.adjust_pitch(30.0)
        # )
        # Create Timers
        self.manual_timer = self.create_timer(
            1/rate,
            self.publish_vel_commands
        )

        self.light_timer = self.create_timer(
            1/10,
            self.send_servo_command
        )

    # Map Twist -> ManualControl
    # x, y, r in [-1000, 1000]
    # z in [0, 1000] with neutral at z_neutral(500)
    def scale_pm1000(self, value, max_value):
        """Map Twist to ManualControl."""
        if max_value <= 1e-6:
            return 0
        return float(round(1000.0 * (value / max_value)))

    def clamp_float(self, v, lo, hi):
        return float(max(lo, min(hi, v)))

    def limit_vel(self, val, limit):
        return max(-limit, min(limit, val))

    def rov_twist_sub(self, msg: Twist):
        """Subscribe to twist messages from user."""
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()

    def send_servo_command(self):
        if not self.servo_cmd:
            return
        # wait for MAVROS service to exist
        if not self.cmd_cli.service_is_ready():
            # don’t spam logs; log occasionally if you want
            return

        if self.adjusting_grip:
            if self.gripper_open:
                pwm = 1900.0
            else:
                pwm = 1100.0
        elif self.adjusting_pitch:
            pwm = self.float_to_pwm(lights=False)

        # Optional: only send if changed
        # TODO Adjust this when lights are set up
        if self.last_sent_pwm is not None and pwm == self.last_sent_pwm:
            self.servo_cmd = False
            return
        if self.adjusting_grip:
            servo_out = int(self.get_parameter('gripper_servo').value)
            self.get_logger().info(f'pwm is {pwm} and servo out is {servo_out}')
        elif self.adjusting_pitch:
            servo_out = int(self.get_parameter('mount_servo').value)

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
        self.servo_cmd = False
        self.adjusting_grip = False
        self.adjusting_pitch = False

    def float_to_pwm(self, lights) -> float:
        pwm_min = int(self.get_parameter('servo_pwm_min').value)
        pwm_max = int(self.get_parameter('servo_pwm_max').value)
        if lights:
            pwm = pwm_min + round(self.brightness * (pwm_max - pwm_min))
        else:
            pwm = pwm_min + round(self.pitch_angle * (pwm_max - pwm_min))
        return max(pwm_min, min(pwm_max, pwm))

    def lights_callback(self, request, response):
        """Toggle the lights on the ROV."""
        if self.brightness == 0.0:
            self.brightness = 1.0
        else:
            self.brightness = 0.0
        self.get_logger().info('Toggling lights')
        return response

    def open_gripper(self, request, response):
        """Service callback to publish servo commands to open the gripper."""
        self.gripper_open = True
        self.adjusting_grip = True
        self.servo_cmd = True
        self.get_logger().info('opening')
        return response

    def close_gripper(self, request, response):
        """Service callback to publish servo commands to open the gripper."""
        self.gripper_open = False
        self.adjusting_grip = True
        self.servo_cmd = True
        return response

    def adjust_pitch(self, angle):
        """Service callback to adjust the camera pitch."""
        self.adjusting_pitch = True
        self.servo_cmd = True
        self.pitch_angle = angle

    def calibrate_bot(self, request, response):
        """Make simple calibration routine, runs of the following commands.

        1. Decend for 1 second
        2. Move right for 1 second
        3. Move left for 1 second
        4. Move forward for 1 second
        5. Move backward for 1 second
        6. Ascend for 1 second

        """
        self.cal_active = True
        self.cal_step = 0
        self.cal_step_start = self.get_clock().now()
        self.get_logger().info('Starting calibration...')
        return response

    def _calibration_twist(self, now):
        if self.cal_step_start is None:
            self.cal_step_start = now

        elapsed = (now - self.cal_step_start).nanoseconds * 1e-9

        # Advance step every 1.0s
        if elapsed >= 2.0:
            self.cal_step += 1
            self.cal_step_start = now
            elapsed = 0.0

        if self.cal_step >= 6:
            self.cal_active = False
            self.get_logger().info('Calibration finished')
            return Twist()  # zero

        z = 0.10
        x = 0.10
        y = 0.10

        cmd = Twist()
        if self.cal_step == 0:
            cmd.linear.z = +z   # ascend
        elif self.cal_step == 1:
            cmd.linear.z = -z   # descend
        elif self.cal_step == 2:
            cmd.linear.x = +x   # forward
        elif self.cal_step == 3:
            cmd.linear.x = -x   # back
        elif self.cal_step == 4:
            cmd.linear.y = +y   # right
        elif self.cal_step == 5:
            cmd.linear.y = -y   # left
        return cmd

    def publish_vel_commands(self):
        """Timer callback: publish MANUAL_CONTROL based on last Twist."""
        now = self.get_clock().now()

        if self.cal_active:
            cmd = self._calibration_twist(now)
        else:
            # Watchdog: if stale, command neutral
            if self.last_cmd_time is None or (now - self.last_cmd_time) > self.cmd_timeout:
                cmd = Twist()
            else:
                cmd = self.last_cmd

        # Clamp incoming Twist
        cmd.linear.x = self.limit_vel(cmd.linear.x, self.max_surge)
        cmd.linear.y = self.limit_vel(cmd.linear.y, self.max_sway)
        cmd.linear.z = self.limit_vel(cmd.linear.z, self.max_heave)
        cmd.angular.z = self.limit_vel(cmd.angular.z, self.max_yaw)

        mc = ManualControl()
        mc.x = self.clamp_float(self.scale_pm1000(cmd.linear.x, self.max_surge), -1000, 1000)
        mc.y = self.clamp_float(self.scale_pm1000(cmd.linear.y, self.max_sway),  -1000, 1000)
        mc.r = self.clamp_float(self.scale_pm1000(cmd.angular.z, self.max_yaw),  -1000, 1000)
        # self.get_logger().info(f'mc is {mc}')

        # heave maps to +/-500 around neutral
        dz = 0.0 if self.max_heave <= 1e-6 else 500.0 * float(cmd.linear.z / self.max_heave)
        mc.z = self.clamp_float(float(self.z_neutral) + dz, 0.0, 1000.0)

        mc.buttons = int(0)   # must be int/uint16

        # Publisher override commands to set light brightness
        pwm = self.float_to_pwm(lights=True)
        ovrd = OverrideRCIn()
        ovrd.channels[10] = pwm
        self.get_logger().info(f'pwm is {pwm}')
        self.lights_pub.publish(ovrd)
        self.manual_pub.publish(mc)


def main():
    """Make the node run."""
    rclpy.init()
    node = Bridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
