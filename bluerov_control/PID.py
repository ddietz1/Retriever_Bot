"""
PID control class to be used by control.py.
"""


class PID:
    """Class with functions for setting PID gains."""

    def __init__(self, kp, ki, kd, dt):
        """Initilize an object with control params."""

        # Parameters
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.dt = dt

        self.integral_err = 0.0  # Integral term
        self.derivative_filtered = 0.0
        self.derivative_alpha = 0.2
        self.derivative_err_last = None

        # --- Integral error testing ---
        self.int_error_cnt = 0

    def update(self, error, forward=None):
        """Update the Twist based on current error."""

        # Proportional(Should be the same as current controller)
        P_term = self.Kp * error # That simple with P

        if abs(error) > 0.05:
            self.integral_err += (error * self.dt)
        
        # NEW: If integral has wrong sign relative to error, decay it fast
        if (error > 0 and self.integral_err < 0) or (error < 0 and self.integral_err > 0):
            # Integral fighting the error direction - decay it
            self.integral_err *= 0.5  # Decay by 50% each cycle

        self.integral_err = max(-1.0, min(1.0, self.integral_err))
        # print(f'integral gain is {self.integral_err}')
        I_term = self.Ki * self.integral_err  # Clamp bounds to prevent overshoot

        # D term requires slope(difference between current and most recent error)
        if self.derivative_err_last is None:
            self.derivative_err_last = error
        derivative_raw = (error - self.derivative_err_last) / self.dt
        self.derivative_filtered = (self.derivative_alpha * derivative_raw + 
                                    (1 - self.derivative_alpha) * self.derivative_filtered)
        D_term = self.Kd * self.derivative_filtered
        self.derivative_err_last = error
        if forward:
            print(f'Terms are P: {P_term:.3f}, D: {D_term:.3f}, I: {I_term:.3f}')

        return max(-1.0, min(1.0, P_term + I_term + D_term))
    
    def reset(self):
        """Reset Integral term to prevent accumulation between runs."""
        self.integral_err = 0.0
        self.derivative_err_last = None
        self.derivative_filtered = 0.0



