"""Module for controlling the ROV velocity.

Overall structure:
Subscribe to /bluerov/ring/object,
Determine ROV velocity based on position from ring
Publish Twist messages to /bluerov/cmd_vel
When close enough, call gripper close service.
"""

from enum import Enum, auto
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from std_srvs.srv import Empty

from sensor_msgs.msg import MagneticField

from bluerov_interfaces.srv import Pitch, Testing
from bluerov_interfaces.msg import Object
from bluerov_control.PID import PID

from mavros_msgs.srv import SetMode

from vision_msgs.msg import Detection2DArray, BoundingBox2D


QOS_IMAGE = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

QOS_DEPTH = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


class State(Enum):
    """Define the state of the system."""

    GRABBING = auto()
    DETECTED = auto()
    SEARCHING = auto()
    IDLE = auto()
    RETRIEVED = auto()
    HOMING = auto()
    TESTING_HORIZONTAL = auto()
    TESTING_VERT = auto()
    TESTING_FORWARD = auto()


class Controller(Node):
    """Controls the ROV based on sensor input."""

    def __init__(self):
        """Initialize necessary variables."""
        super().__init__('controller')

        # State
        self.State = State.IDLE

        # Declare params for PID control constants
        self.declare_parameter("PID_forward.kp", 0.132)
        self.declare_parameter("PID_forward.ki", 0.0241) # 0.025
        self.declare_parameter("PID_forward.kd", 0.025) # 0.025

        self.declare_parameter("PID_vertical.kp", -0.2) # 0.17
        self.declare_parameter("PID_vertical.ki", -0.005) # 0.005
        self.declare_parameter("PID_vertical.kd", 0.0) # 0.051

        self.declare_parameter("PID_horizontal.kp", 0.1) #0.17
        self.declare_parameter("PID_horizontal.ki", 0.001) # 0.01
        self.declare_parameter("PID_horizontal.kd", 0.005) # 0.02

        self.declare_parameter("PID_heading.kp", 0.14) # 0.16
        self.declare_parameter("PID_heading.ki", 0.001)
        self.declare_parameter("PID_heading.kd", 0.005) # 0.015

        # Control constants (tune)
        # --- Trying with just dt = 0.05
        self.dt = 0.04  # Controller publishes roughly 25 fps
        self.PID_forward = PID(
            self.get_parameter("PID_forward.kp").value,
            self.get_parameter("PID_forward.ki").value,
            self.get_parameter("PID_forward.kd").value,
            self.dt
        )

        self.PID_vertical = PID(
            self.get_parameter("PID_vertical.kp").value,
            self.get_parameter("PID_vertical.ki").value,
            self.get_parameter("PID_vertical.kd").value,
            self.dt
        )

        self.PID_horizontal = PID(
            self.get_parameter("PID_horizontal.kp").value,
            self.get_parameter("PID_horizontal.ki").value,
            self.get_parameter("PID_horizontal.kd").value,
            self.dt
        )

        self.PID_heading = PID(
            self.get_parameter("PID_heading.kp").value,
            self.get_parameter("PID_heading.ki").value,
            self.get_parameter("PID_heading.kd").value,
            self.dt
        )

        self.eps = 0.02

        # --- Heading hold parameters ---
        self.Kp_heading = 0.5  # Proportional gain for heading control (tune)
        self.search_forward_speed = 0.0  # Forward speed during search (tune)
        self.current_heading = None  # Current heading from magnetometer
        self.target_heading = None  # Target heading to maintain
        self.heading_initialized = False

        # --- "size" thresholds (msg.area) ---
        self.target_size = 0.68  # close enough to grab (tune in water)
        self.far_size = 0.2    # far/close transition (tune)
        # self.close_size = 0.18  # close/near transition (tune)

        # Gripper alignment / grab gating
        self.gripper_offset_x = 0.1  # Tested experimentally
        self.grab_ex_tol = 0.25
        self.grab_ey_tol = 0.6
        self.grab_frames_required = 5
        self.within_reach_counter = 0

        # Depth hold (tune)
        self.Kp_depth = 0.12
        self.vz_max = 0.1
        self.y_on = 0.05
        self.y_off = 0.10
        self.depth_lock_frames = 10

        # Timers / counters
        self.grabbing_timer = 0
        self.checking_timer = 0
        self.detected_timer = 0
        self.detected_ring_timer = 0
        self.depth_lock_counter = 0

        # Test Counters
        self.test_forward_count = 0
        self.test_strafe_count = 0
        self.test_heave_count = 0

        # Flags
        self.gripper_closed = False
        self.pitch_set_search = False
        self.pitch_set_check = False
        self.lights_on = False
        self.depth_hold_enabled = False

        # Testing flags
        self.test_timer_started = False
        self.test_forward_timer = None
        self.test_strafe_timer = None
        self.test_heave_timer = None
        self.test_initial_error = None

        # Logging throttle
        self.log_every_n = 4
        self.log_count = 0

        # Depth state
        self.depth = None
        self.depth_setpoint = 0.0

        # Searching logic
        self.search_timer_start = None
        self.search_forward_duration = 26
        self.search_side_duration = 6
        # Set these from tests in pool
        self.forward_heading = 3.75 # 4.6 from raw msg
        self.right_heading = 3.35
        self.backward_heading = 2.215
        self.left_heading = 0.078

        self.search_forward_speed = 0.08
        self.search_sideways_speed = 0.01 # drift will add to this, tune

        self.init_decend = True
        self.drift_offset = 0.04 # Test this

        # Yolo params
        self.detect_cnt = 0
        self.yolo_target_size = 0.7

        # Callback group for service clients
        self.cb = MutuallyExclusiveCallbackGroup()

        # Subscribers
        self.object_sub = self.create_subscription(
            Object,
            '/bluerov/ring/object',
            self.object_cb,
            QOS_IMAGE,
        )

        self.depth_sub = self.create_subscription(
            Float64,
            '/mavros/global_position/rel_alt',
            self.get_alt,
            QOS_DEPTH,
        )

        # self.mag_sub = self.create_subscription(
        #     MagneticField,
        #     '/mavros/imu/mag',
        #     self.mag_callback,
        #     QOS_DEPTH
        # )

        self.heading_sub = self.create_subscription(
            Float64,
            '/rov/heading',
            self.heading_cb,
            QOS_DEPTH
        )

        self.yolo_sub = self.create_subscription(
            Detection2DArray,
            '/boundingBox',
            self.yolo_cb,
            QOS_IMAGE
        )

        # Publisher
        self.twist_pub = self.create_publisher(Twist, '/bluerov/cmd_vel', 10)

        # Services
        # Service server to start detection
        self.run_detection = self.create_service(
            Empty,
            '/bluerov/detect',
            self.toggle_detection,
        )

        self.test_forward = self.create_service(
            Testing,
            "test",
            self.toggle_testing
        )

        self.reset_test = self.create_service(
            Empty,
            'reset_test',
            self.reset_test_callback
        )

        # Service clients
        self.pitch_client = self.create_client(
            Pitch,
            '/bluerov/adjust_pitch',
            callback_group=self.cb,
        )
        self.lights_client = self.create_client(
            Empty,
            '/bluerov/set_lights',
            callback_group=self.cb,
        )
        self.close_grip_client = self.create_client(
            Empty,
            '/bluerov/close_gripper',
            callback_group=self.cb,
        )
        self.open_grip_client = self.create_client(
            Empty,
            '/bluerov/open_gripper',
            callback_group=self.cb,
        )

        self.change_mode = self.create_client(
            SetMode,
            '/mavros/set_mode',
            callback_group=self.cb,
        )
        self.startup_timer = self.create_timer(2.0, self._startup_mode_reset)

    # ---- Helper Functions ---- #

    def _startup_mode_reset(self):
        """Ensure we start in MANUAL mode, not ALT_HOLD."""
        self.change_mode_('MANUAL')
        self.get_logger().info('Startup mode reset to MANUAL')
        # One-shot timer so we destroy it after firing
        self.startup_timer.cancel()
    def yolo_cb(self, msg: Detection2DArray):
        '''Callback for yolo msgs.'''
        # self.get_logger().info(f'callback!')
        # Filter to detections above confidence threshold
        valid_detections = [
            d for d in msg.detections
            if d.bbox.size_x > 0 and d.bbox.size_y > 0
        ]
        detect = len(valid_detections) > 0
        if detect:
            self.detect_cnt += 1
            #self.get_logger().info('detected one')
        else:
            self.detect_cnt = 0

        detected = self.detect_cnt > 20
        if detect:
            bbox = valid_detections[0].bbox  # Use first valid detection
            raw_x = bbox.center.position.x
            raw_y = bbox.center.position.y
            size = (bbox.size_x / 640.0) * (bbox.size_y / 480.0)

            center_x = (raw_x - (1280 / 2.0)) / (1280 / 2.0)
            center_y = (raw_y - (720 / 2.0)) / (720 / 2.0)
        else:
            center_x = 0.0
            center_y = 0.0
            size = 0.0
        self.controller(center_x, center_y, detected, size, yolo=True)

    def object_cb(self, msg: Object):
        '''Callback for Object msgs.'''
        center_x = float(msg.cx)          # +right in camera frame
        center_y = float(msg.cy)          # +down in camera frame
        detected = bool(msg.detected)
        size = float(msg.area)
        self.controller(center_x, center_y, detected, size, yolo=False)

    def _service_cb(self, label: str):
        def _done_cb(fut):
            try:
                fut.result()  # raises on failure
                self.get_logger().info(f'{label} call succeeded')
            except Exception as e:
                self.get_logger().error(f'{label} call failed: {e}')
        return _done_cb

    def _depth_hold(self) -> float:
        """Return z command to hold depth_setpoint."""
        if (self.depth is None) or (not self.depth_hold_enabled):
            return 0.0
        # Typical convention: rel_alt increases as you go deeper.
        # If your z sign is inverted in your bridge, flip this sign.
        err = self.depth_setpoint - self.depth
        vz = self.Kp_depth * err
        return max(-self.vz_max, min(self.vz_max, vz))
    
    def heading_cb(self, msg: Float64):
        """Gets heading from onboard magnetometer"""
        # Need to test in water to determine the proper headings for NSEW
        self.current_heading = msg.data
        # self.get_logger().info(f'data is {self.current_heading}')

    def _apply_x_offset(self, x: float) -> float:
        """Once the ring is close, adjust x offset to ensure alignment."""
        return x - self.gripper_offset_x

    # def mag_callback(self, msg: MagneticField):
    #     """Calculate heading from magnetometer readings."""
    #     mag_x = msg.magnetic_field.x
    #     mag_y = msg.magnetic_field.y

    #     # Calculate heading in degrees (0-360)
    #     heading_rad = math.atan2(mag_y, mag_x)
    #     heading_deg = math.degrees(heading_rad)

    #     # Normalize to 0-360
    #     if heading_deg < 0:
    #         heading_deg += 360

    #     self.current_heading = heading_deg

    #     # Initialize target heading on first reading
    #     if not self.heading_initialized and self.current_heading is not None:
    #         self.target_heading = self.current_heading
    #         self.heading_initialized = True
    #         self.get_logger().info(f'Heading initialized to {self.target_heading:.1f}°')

    def heading_controller(self, current_heading: float, target_heading: float) -> float:
        """
        Simple proportional controller for heading.
        Returns yaw rate command (-1 to 1).
        """
        # Calculate error with wrap-around handling
        error = (target_heading - current_heading + math.pi) % (2 * math.pi) - math.pi
    
        if abs(error) > 0.05:  # ~3 degrees in radians
            yaw_command = self.PID_heading.update(error)
            yaw_command = max(-0.2, min(0.2, yaw_command))
        else:
            yaw_command = 0.0
        #self.get_logger().info(f'heading error={error:.3f} rad, yaw_cmd={yaw_command:.3f}')
        
        return -yaw_command


    def get_alt(self, msg: Float64):
        """Track current depth (rel_alt)."""
        self.depth = float(msg.data)

    # --- Service callbacks ---
    def toggle_testing(self, request, response):
        """Toggle testing state"""
        self.test = request.type.lower()
        if self.test == 'forward':
            self._enter_state(State.TESTING_FORWARD)
        elif self.test == 'horizontal':
            self._enter_state(State.TESTING_HORIZONTAL)
        elif self.test == 'vertical':
            self._enter_state(State.TESTING_VERT)
        return response
    
    def reset_test_callback(self, request, response):
        self.test_timer_started = False
        self.test_forward_count = 0
        self.test_strafe_count = 0
        self.test_heave_count = 0
        self._enter_state(State.IDLE)
        return response

    def change_mode_(self, mode):
        """Service to change mode(ALT_HOLD or MANUAL)."""
        if not self.change_mode.service_is_ready():
            self.get_logger().info('ALT_HOLD service is not ready')
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = mode
        future = self.change_mode.call_async(req)
        future.add_done_callback(self.change_mode_response)

    def change_mode_response(self, future):
        """Callback from change mode_ service."""
        try:
            response = future.result()
            if response.mode_sent:
                self.get_logger().info("Mode change successful")
            else:
                self.get_logger().error("Mode change failed")
        except Exception as e:
            self.get_logger().error(f"Service call failed {e}")

    def call_pitch_service(self, degrees: float):
        """Call pitch service to rotate the camera mount."""
        if not self.pitch_client.service_is_ready():
            self.get_logger().warn('Pitch service not ready')
            return
        req = Pitch.Request()
        req.angle = float(degrees)
        future = self.pitch_client.call_async(req)
        future.add_done_callback(self._service_cb(f'Pitch({degrees:.1f})'))

    def call_lights_service(self):
        """Call lights service to turn on lights."""
        if not self.lights_client.service_is_ready():
            self.get_logger().warn('Lights service not ready')
            return
        req = Empty.Request()
        future = self.lights_client.call_async(req)
        future.add_done_callback(self._service_cb('Lights'))

    def call_grip_client(self, open_: bool):
        """Call either the close or open gripper service."""
        req = Empty.Request()
        if open_:
            if not self.open_grip_client.service_is_ready():
                self.get_logger().warn('Open gripper service not ready')
                return
            future = self.open_grip_client.call_async(req)
            future.add_done_callback(self._service_cb('Open Gripper'))
            self.gripper_closed = False
        else:
            if not self.close_grip_client.service_is_ready():
                self.get_logger().warn('Close gripper service not ready')
                return
            if self.gripper_closed:
                return
            future = self.close_grip_client.call_async(req)
            future.add_done_callback(self._service_cb('Close Gripper'))
            self.gripper_closed = True

    def _enter_state(self, new_state: State) -> None:
        if new_state == self.State:
            return
        self.get_logger().info(f'State: {self.State.name} -> {new_state.name}')
        self.State = new_state

        # Reset per-state counters / flags
        if new_state == State.GRABBING:
            self.grabbing_timer = 0
        if new_state == State.RETRIEVED:
            self.checking_timer = 0
            self.pitch_set_check = False
        if new_state == State.SEARCHING:
            # Start timer for lawnmower search logic
            self.search_timer_start = None
            # stop holding depth while searching
            self.depth_hold_enabled = False
            self.depth_lock_counter = 0
        if new_state == State.DETECTED:
            # Disable depth hold
            self.change_mode_('MANUAL')
        if new_state == State.IDLE:
            self.twist_pub.publish(Twist())
            self.change_mode_('MANUAL')

        # Reset PID terms
        self.PID_forward.reset()
        self.PID_vertical.reset()
        self.PID_horizontal.reset()
        self.PID_heading.reset()

    def toggle_detection(self, request, response):
        """Start searching and arm lights/pitch once."""
        self._enter_state(State.SEARCHING)

        if not self.pitch_set_search:
            self.call_pitch_service(-25.0)
            self.pitch_set_search = True

        if not self.lights_on:
            # self.call_lights_service()
            self.lights_on = True

        self.call_grip_client(open_=True)
        return response

    def controller(self, center_x, center_y, detected, size, yolo):
        """Control the ROV based on object detection."""
        # Call control node
        # IMPORTANT: size is a filtered close-range size metric in [0,1]
        if yolo:
            target_size = self.yolo_target_size    # tune in pool tonight
        else:
            target_size = self.target_size

        # Always define diffs so deadband at end can't crash
        diff_x = center_x
        diff_y = center_y

        # State transition SEARCHING <-> DETECTED
        if detected and (self.State == State.SEARCHING):
            self._enter_state(State.DETECTED)
            self.detected_timer = 0
            self.within_reach_counter = 0
        elif (not detected) and (self.State == State.DETECTED):
            if self.detected_timer < 101:
                self.detected_timer += 1
            else:
                self._enter_state(State.SEARCHING)
                self.within_reach_counter = 0
        else:
            self.detected_timer = 0

        tw = Twist()  # defaults to zero

        # --- TESTING ---

        # Testing heading control for lawnmover search

        if self.State == State.TESTING_FORWARD:
            # Start a timer once the testing has started
            if not self.test_timer_started:
                self.test_timer_started = True
                self.test_forward_timer = self.get_clock().now()
                self.test_forward_count = 0
                self.test_initial_size = size
                self.get_logger().info(f'Starting forward test: initial_size={size:.3f}')

            diff_size = 0.85 - size  # Setting this to 0.6 to see what the overshoot is like

            tw.linear.x = self.PID_forward.update(diff_size, forward=True)
            tw.linear.y = self.PID_horizontal.update(diff_x)  # Keep centered horizontally
            tw.linear.z = self.PID_vertical.update(diff_y) # Keep centered vertically
            tw.angular.z = self.PID_horizontal.update(diff_x)

            if abs(diff_size) < 0.1:
                self.test_forward_count += 1
            else:
                self.test_forward_count = 0

            # Log progress periodically
            if self.test_forward_count % 10 == 0 and self.test_forward_count > 0:
                elapsed = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
                self.get_logger().info(
                    f'Forward test: t={elapsed:.1f}s, error={diff_size:.3f}, '
                    f'stable={self.test_forward_count}/20'
                )
            
            # Test complete
            if self.test_forward_count >= 20:
                total_time = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
                self.get_logger().info(
                    f'Forward test COMPLETE: settling_time={total_time:.2f}s, '
                    f'initial_error={0.6-self.test_initial_size:.3f}, '
                    f'final_error={diff_size:.3f}'
                )
                # Reset and return to idle
                self.test_timer_started = False
                self.test_forward_count = 0
                self._enter_state(State.IDLE)
            
            # Timeout check
            elapsed = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
            if elapsed > 45.0:
                self.get_logger().warn(f'Forward test TIMEOUT after {elapsed:.1f}s')
                self.test_timer_started = False
                self._enter_state(State.IDLE)

        if self.State == State.TESTING_HORIZONTAL:
            # Start a timer once the testing has started
            if not self.test_timer_started:
                self.test_timer_started = True
                self.test_forward_timer = self.get_clock().now()
                self.test_forward_count = 0
                self.test_initial_err = diff_x
                self.get_logger().info(f'Starting forward test: initial_size={diff_x:.3f}')

            diff_size = 0.6 - size  # Setting this to 0.6 to see what the overshoot is like

            tw.linear.x = self.PID_forward.update(diff_size, forward=True)
            tw.linear.y = self.PID_horizontal.update(diff_x)
            tw.linear.z = self.PID_vertical.update(diff_y) # Keep centered vertically
            tw.angular.z = self.PID_horizontal.update(diff_x)

            if abs(diff_x) < 0.05:
                self.test_forward_count += 1
            else:
                self.test_forward_count = 0

            # Log progress periodically
            if self.test_forward_count % 10 == 0 and self.test_forward_count > 0:
                elapsed = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
                self.get_logger().info(
                    f'Forward test: t={elapsed:.1f}s, error={diff_x:.3f}, '
                    f'stable={self.test_forward_count}/20'
                )
            
            # Test complete
            if self.test_forward_count >= 20:
                total_time = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
                self.get_logger().info(
                    f'Forward test COMPLETE: settling_time={total_time:.2f}s, '
                    f'initial_error={0.6-self.test_initial_err:.3f}, '
                    f'final_error={diff_x:.3f}'
                )
                # Reset and return to idle
                self.test_timer_started = False
                self.test_forward_count = 0
                self._enter_state(State.IDLE)
            
            # Timeout check
            elapsed = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
            if elapsed > 45.0:
                self.get_logger().warn(f'Forward test TIMEOUT after {elapsed:.1f}s')
                self.test_timer_started = False
                self._enter_state(State.IDLE)

        if self.State == State.TESTING_VERT:
            # Start a timer once the testing has started
            if not self.test_timer_started:
                self.test_timer_started = True
                self.test_forward_timer = self.get_clock().now()
                self.test_forward_count = 0
                self.test_initial_err = diff_y
                self.get_logger().info(f'Starting forward test: initial_size={diff_y:.3f}')

            diff_size = 0.6 - size  # Setting this to 0.6 to see what the overshoot is like

            tw.linear.x = self.PID_forward.update(diff_size)
            tw.linear.z = self.PID_vertical.update(diff_y)
            tw.linear.y = self.PID_horizontal.update(diff_x)  # Keep centered horizontally
            tw.angular.z = self.PID_horizontal.update(diff_x)

            if abs(diff_y) < 0.1:
                self.test_forward_count += 1
            else:
                self.test_forward_count = 0

            # Log progress periodically
            if self.test_forward_count % 10 == 0 and self.test_forward_count > 0:
                elapsed = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
                self.get_logger().info(
                    f'Forward test: t={elapsed:.1f}s, error={diff_y:.3f}, '
                    f'stable={self.test_forward_count}/20'
                )
            
            # Test complete
            if self.test_forward_count >= 20:
                total_time = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
                self.get_logger().info(
                    f'Forward test COMPLETE: settling_time={total_time:.2f}s, '
                    f'initial_error={0.6-self.test_initial_size:.3f}, '
                    f'final_error={diff_y:.3f}'
                )
                # Reset and return to idle
                self.test_timer_started = False
                self.test_forward_count = 0
                self._enter_state(State.IDLE)
            
            # Timeout check
            elapsed = (self.get_clock().now() - self.test_forward_timer).nanoseconds / 1e9
            if elapsed > 45.0:
                self.get_logger().warn(f'Forward test TIMEOUT after {elapsed:.1f}s')
                self.test_timer_started = False
                self._enter_state(State.IDLE)

        # --- SEARCHING ---
        if self.State == State.SEARCHING:
            heading = self.forward_heading
            if self.search_timer_start is None:
                self.search_timer_start = self.get_clock().now()
            total_time = (self.get_clock().now() - self.search_timer_start).nanoseconds / 1e9
            # Move forward
            if total_time < self.search_forward_duration:
                tw.linear.y = -self.drift_offset
                # Decend while moving forward
                if self.init_decend:
                    tw.linear.z = -0.2
                tw.linear.x = self.search_forward_speed
            # Move right
            elif total_time < (self.search_forward_duration + self.search_side_duration):
                if self.init_decend:
                    # Set depth hold
                    self.change_mode_('ALT_HOLD')
                    self.init_decend = False
                tw.linear.y = self.search_sideways_speed
            # Move back
            elif total_time < (2*self.search_forward_duration + self.search_side_duration):
                tw.linear.y = -self.drift_offset
                tw.linear.x = -self.search_forward_speed
            
            elif total_time < (2*self.search_forward_duration + 2*self.search_side_duration):
                tw.linear.y = self.search_sideways_speed
            else:
                # Reset timer and start again
                self.search_timer_start = self.get_clock().now()
                return

            if self.current_heading is None:
                self.get_logger().info("There is no current heading")
                return
            if heading is None:
                self.get_logger().info("There is no target heading")
                return
            yaw = self.heading_controller(self.current_heading, heading)
            tw.angular.z = yaw  # Keep an even keel

        # --- DETECTED ---
        elif self.State == State.DETECTED:
            if detected and (self.depth is not None):
                # Enter depth hold after |diff_y| is 'good' for N frames
                if (not self.depth_hold_enabled) and (abs(diff_y) < self.y_on):
                    self.depth_lock_counter += 1
                    if self.depth_lock_counter >= self.depth_lock_frames:
                        self.depth_hold_enabled = True
                        self.depth_setpoint = self.depth
                else:
                    if not self.depth_hold_enabled:
                        self.depth_lock_counter = 0

                # Exit depth hold if we drift vertically
                if self.depth_hold_enabled and (abs(diff_y) > self.y_off):
                    self.depth_hold_enabled = False
                    self.depth_lock_counter = 0
            else:
                self.depth_hold_enabled = False
                self.depth_lock_counter = 0

            # Decide z command
            # if self.depth_hold_enabled:
            #     tw.linear.z = self._depth_hold()
            tw.linear.z = self.PID_vertical.update(diff_y)

            # Approach error: want size to increase to target_size
            diff_size = max(0.0, target_size - size)

            # --- Approach logic based on stable size metric ---
            if size < 0.65:  # near: align to gripper offset, changing to 0.8 from self.target_size 
                tw.linear.y = self.PID_horizontal.update(self._apply_x_offset(diff_x))  # Removing x offset for now
                tw.linear.x = self.PID_forward.update(diff_size, forward=True)
                tw.angular.z = self.PID_horizontal.update(diff_x)
                self.within_reach_counter = 0

            else:  # size >= target_size => candidate grab zone
                centered = (abs(diff_x - 0.1) < self.grab_ex_tol) and (abs(diff_y) < self.grab_ey_tol)

                if centered:
                    # Debounce grab condition for N frames
                    if self.within_reach_counter < self.grab_frames_required:
                        self.within_reach_counter += 1
                    else:
                        self._enter_state(State.GRABBING)
                        self.within_reach_counter = 0
                else:
                    # Not centered: keep trying to center
                    self.within_reach_counter = 0
                tw.linear.y = self.PID_horizontal.update(self._apply_x_offset(diff_x))
                tw.linear.x = self.PID_forward.update(diff_size)
                tw.angular.z = self.PID_horizontal.update(diff_x)

        # --- GRABBING ---
        elif self.State == State.GRABBING:
            diff_size = max(0.0, target_size - size)
            if self.grabbing_timer < 10:
                tw.linear.x = self.PID_forward.update(diff_size) + 0.055  # Keep moving forward, will slip otherwise
                self.grabbing_timer += 1
            else:
                self.call_grip_client(open_=False)
                self._enter_state(State.RETRIEVED)

        # --- RETRIEVED ---
        elif self.State == State.RETRIEVED:
            if self.checking_timer < 45:
                tw.linear.x = -0.05
                self.checking_timer += 1
            else:
                if not self.pitch_set_check:
                    self.call_pitch_service(-85.0)
                    self.pitch_set_check = True
                if detected:
                    # Start detected timer
                    if self.detected_ring_timer > 60:
                        self._enter_state(State.HOMING)  # Done
                    else:
                        self.detected_ring_timer += 1
                else:
                    # Reduce timer
                    if self.detected_ring_timer == 0:
                        # Reenter search mode
                        self.get_logger().info('Ring no longer detected, reentering search mode')
                        self._enter_state(State.SEARCHING)
                        self.call_pitch_service(-30.0)
                        self.pitch_set_check = False
                        self.call_grip_client(open_=True)
                    else:
                        self.detected_ring_timer -= 1

        # --- HOMING ---
        elif self.State == State.HOMING:
            tw.linear.z = -0.03  # Slowly ascend

        # ---- Throttled logging (never gates behavior) ----
        self.log_count += 1
        if self.log_count % self.log_every_n == 0:
            self.get_logger().info(
                f'state={self.State.name} size={size:.3f} ex={diff_x:+.2f} ey={diff_y:+.2f}'
                f'x: {tw.linear.x}, y: {tw.linear.y}, z: {tw.linear.z}, ang_z: {tw.angular.z}'
            )

        # ---- Apply deadbands just before publishing ----
        if (abs(diff_x) < self.eps) and self.State != State.SEARCHING:
            # self.get_logger().info("deadband hit")
            tw.angular.z = 0.0
            tw.linear.y = 0.0

        if (not self.depth_hold_enabled) and (abs(diff_y) < self.eps):
            tw.linear.z = 0.0
        if self.State != State.IDLE:
            self.twist_pub.publish(tw)



def main():
    """Run the ROS2 node."""
    rclpy.init()
    node = Controller()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
