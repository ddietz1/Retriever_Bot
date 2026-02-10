"""Module for controlling the ROV velocity.

Overall structure:
Subscribe to /bluerov/ring/object,
Determine ROV velocity based on position from ring
Publish Twist messages to /bluerov/cmd_vel
When close enough, call gripper close service

NOTE:
- In this version, msg.area is treated as a *stable size metric* in [0, 1]
  (e.g., filtered enclosing-circle radius from your detector), not raw contour area.
"""

from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from std_srvs.srv import Empty

from bluerov_interfaces.srv import Pitch
from bluerov_interfaces.msg import Object


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
    RING_DETECTED = auto()
    SEARCHING = auto()
    IDLE = auto()
    RETRIEVED = auto()
    HOMING = auto()


class Controller(Node):
    """Controls the ROV based on sensor input."""

    def __init__(self):
        """Initialize necessary variables."""
        super().__init__('controller')

        # State
        self.State = State.IDLE

        # Control constants (tune)
        self.eps = 0.05
        self.P_yaw = 0.16
        self.P_heave = -0.12
        self.P_forward = 0.2
        self.P_strafe = 0.16

        # --- UPDATED: "size" thresholds (msg.area) ---
        # Treat msg.area as "size_norm" in [0,1] (stable close metric)
        self.target_size = 0.95   # close enough to grab (tune in water)
        self.far_size = 0.45      # far/close transition (tune)
        self.close_size = 0.65    # close/near transition (tune)

        # Gripper alignment / grab gating
        self.gripper_offset_x = 0.2  # Tested experimentally
        self.grab_ex_tol = 0.45
        self.grab_ey_tol = 0.45
        self.grab_frames_required = 10
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

        # Flags
        self.gripper_closed = False
        self.pitch_set_search = False
        self.pitch_set_check = False
        self.lights_on = False
        self.depth_hold_enabled = False

        # Logging throttle
        self.log_every_n = 40
        self.log_count = 0

        # Depth state
        self.depth = None
        self.depth_setpoint = 0.0

        # Callback group for service clients
        self.cb = MutuallyExclusiveCallbackGroup()

        # Subscribers
        self.object_sub = self.create_subscription(
            Object,
            '/bluerov/ring/object',
            self.controller,
            QOS_IMAGE,
        )

        self.depth_sub = self.create_subscription(
            Float64,
            '/mavros/global_position/rel_alt',
            self.get_alt,
            QOS_DEPTH,
        )

        # Publisher
        self.twist_pub = self.create_publisher(Twist, '/bluerov/cmd_vel', 10)

        # Service server to start detection
        self.run_detection = self.create_service(
            Empty,
            '/bluerov/detect',
            self.toggle_detection,
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

    # ---- Helper Functions ---- #
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

    def _apply_x_offset(self, x: float) -> float:
        """Once the ring is close, adjust x offset to ensure alignment."""
        return x - self.gripper_offset_x

    def get_alt(self, msg: Float64):
        """Track current depth (rel_alt)."""
        self.depth = float(msg.data)

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
            # stop holding depth while searching
            self.depth_hold_enabled = False
            self.depth_lock_counter = 0

    def toggle_detection(self, request, response):
        """Start searching and arm lights/pitch once."""
        self._enter_state(State.SEARCHING)

        if not self.pitch_set_search:
            self.call_pitch_service(-40.0)
            self.pitch_set_search = True

        if not self.lights_on:
            # self.call_lights_service()
            self.lights_on = True

        self.call_grip_client(open_=True)
        return response

    def controller(self, msg: Object):
        """Control the ROV based on object detection."""
        center_x = float(msg.cx)          # +right in camera frame
        center_y = float(msg.cy)          # +down in camera frame

        # IMPORTANT: size is a filtered close-range size metric in [0,1]
        size = float(msg.area)

        detected = bool(msg.detected)
        circle_percent = float(msg.circularity)

        # Always define diffs so deadband at end can't crash
        diff_x = center_x
        diff_y = center_y

        # State transition SEARCHING <-> RING_DETECTED
        if detected and (self.State == State.SEARCHING):
            self._enter_state(State.RING_DETECTED)
            self.detected_timer = 0
            self.within_reach_counter = 0
        elif (not detected) and (self.State == State.RING_DETECTED):
            if self.detected_timer < 101:
                self.detected_timer += 1
            else:
                self._enter_state(State.SEARCHING)
                self.within_reach_counter = 0
        else:
            self.detected_timer = 0

        tw = Twist()  # defaults to zero

        # --- SEARCHING ---
        if self.State == State.SEARCHING:
            tw.angular.z = 0.0

        # --- RING_DETECTED ---
        elif self.State == State.RING_DETECTED:
            # ---- Depth-hold latch logic (ONLY in RING_DETECTED) ----
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
            if self.depth_hold_enabled:
                tw.linear.z = self._depth_hold()
            else:
                tw.linear.z = self.P_heave * diff_y

            # Approach error: want size to increase to target_size
            diff_size = max(0.0, self.target_size - size)

            # --- Approach logic based on stable size metric ---
            if 0.0 < size < self.far_size:  # far
                tw.angular.z = self.P_yaw * diff_x
                tw.linear.x = self.P_forward * diff_size

            elif self.far_size <= size < self.close_size:  # close
                tw.linear.y = self.P_strafe * diff_x
                tw.linear.x = self.P_forward * diff_size

            elif self.close_size <= size < self.target_size:  # near: align to gripper offset
                tw.linear.y = self.P_strafe * self._apply_x_offset(diff_x)
                tw.linear.x = self.P_forward * diff_size
                self.within_reach_counter = 0

            else:  # size >= target_size => candidate grab zone
                centered = (abs(diff_x - 0.2) < self.grab_ex_tol) and (abs(diff_y) < self.grab_ey_tol)

                if centered:
                    # Debounce grab condition for N frames
                    if self.within_reach_counter < self.grab_frames_required:
                        self.within_reach_counter += 1
                    else:
                        self._enter_state(State.GRABBING)
                        self.within_reach_counter = 0
                else:
                    # Not centered: keep trying to center, but do NOT keep pushing forward
                    self.within_reach_counter = 0
                    tw.linear.y = self.P_strafe * self._apply_x_offset(diff_x)
                    tw.linear.x = 0.0
                    tw.angular.z = self.P_yaw * diff_x

        # --- GRABBING ---
        elif self.State == State.GRABBING:
            if self.grabbing_timer < 15:
                tw.linear.x = 0.05
                self.grabbing_timer += 1
            else:
                self.call_grip_client(open_=False)
                self._enter_state(State.RETRIEVED)

        # --- RETRIEVED ---
        elif self.State == State.RETRIEVED:
            if self.checking_timer < 15:
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
                f'state={self.State.name} size={size:.3f} ex={diff_x:+.2f} ey={diff_y:+.2f} '
                f'circ={circle_percent:.2f} depth={self.depth} hold={self.depth_hold_enabled} '
                f'x: {tw.linear.x}, y: {tw.linear.y}, z: {tw.linear.z}, ang_z: {tw.angular.z}'
            )

        # ---- Apply deadbands just before publishing ----
        if abs(diff_x) < self.eps:
            tw.angular.z = 0.0
            tw.linear.y = 0.0

        if (not self.depth_hold_enabled) and (abs(diff_y) < self.eps):
            tw.linear.z = 0.0

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
