import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL
from opendbc.car.common.pid import PIDController
from opendbc.car.body import bodycan
from opendbc.car.body.values import SPEED_FROM_RPM
from opendbc.car.interfaces import CarControllerBase

MAX_TORQUE = 500
MAX_TORQUE_RATE = 50
MAX_ANGLE_ERROR = np.radians(7)
MAX_POS_INTEGRATOR = 0.2   # meters
MAX_TURN_INTEGRATOR = 0.1  # meters

# limits on how fast the speed targets may change, so a step on the joystick
# becomes a ramp (a trapezoid instead of a box). slowing down is allowed to be quicker
MAX_ACCEL = 1.0        # m/s^2
MAX_DECEL = 2.5        # m/s^2
MAX_TURN_ACCEL = 2.0   # m/s^2, wheel speed difference
MAX_TURN_DECEL = 4.0   # m/s^2

TORQUE_DEADBAND = 10
TORQUE_DEADBAND_BLEND = 5  # torque over which the deadband compensation fades in


def rate_limit(target: float, current: float, accel: float, decel: float) -> float:
  # speeding up means moving away from zero; anything else (including reversing) is slowing down
  slowing = abs(target) < abs(current) or (current != 0. and np.sign(target) != np.sign(current))
  step = (decel if slowing else accel) * DT_CTRL
  return float(np.clip(target, current - step, current + step))


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.main])

    # PIDs
    self.turn_pid = PIDController(110, k_i=11.5, rate=1 / DT_CTRL)
    self.wheeled_speed_pid = PIDController(110, k_i=11.5, rate=1 / DT_CTRL)

    self.speed_desired = 0.
    self.speed_diff_desired = 0.
    self.torque_r_filtered = 0.
    self.torque_l_filtered = 0.

  @staticmethod
  def deadband_filter(torque, deadband):
    # fade the compensation in around zero. a hard +/- deadband flips sign with
    # the noise when the torque is near zero, which chatters the motors at rest
    return torque + deadband * float(np.clip(torque / TORQUE_DEADBAND_BLEND, -1., 1.))

  def update(self, CC, CS, now_nanos):

    torque_l = 0
    torque_r = 0

    if CC.enabled:
      # Read these from the joystick
      # TODO: this isn't acceleration, okay?
      self.speed_desired = rate_limit(CC.actuators.accel / 5., self.speed_desired, MAX_ACCEL, MAX_DECEL)
      self.speed_diff_desired = rate_limit(-CC.actuators.torque / 2., self.speed_diff_desired, MAX_TURN_ACCEL, MAX_TURN_DECEL)

      speed_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr) / 2.
      speed_error = self.speed_desired - speed_measured

      torque = self.wheeled_speed_pid.update(speed_error, freeze_integrator=False)

      speed_diff_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)
      turn_error = speed_diff_measured - self.speed_diff_desired
      freeze_integrator = ((turn_error < 0 and self.turn_pid.error_integral <= -MAX_TURN_INTEGRATOR) or
                           (turn_error > 0 and self.turn_pid.error_integral >= MAX_TURN_INTEGRATOR))
      torque_diff = self.turn_pid.update(turn_error, freeze_integrator=freeze_integrator)

      # Combine 2 PIDs outputs
      torque_r = torque + torque_diff
      torque_l = torque - torque_diff

      # Torque rate limits
      self.torque_r_filtered = np.clip(self.deadband_filter(torque_r, TORQUE_DEADBAND),
                                       self.torque_r_filtered - MAX_TORQUE_RATE,
                                       self.torque_r_filtered + MAX_TORQUE_RATE)
      self.torque_l_filtered = np.clip(self.deadband_filter(torque_l, TORQUE_DEADBAND),
                                       self.torque_l_filtered - MAX_TORQUE_RATE,
                                       self.torque_l_filtered + MAX_TORQUE_RATE)
      torque_r = int(np.clip(self.torque_r_filtered, -MAX_TORQUE, MAX_TORQUE))
      torque_l = int(np.clip(self.torque_l_filtered, -MAX_TORQUE, MAX_TORQUE))
    else:
      # start from rest when re-enabled instead of from stale targets and integrators
      self.speed_desired = 0.
      self.speed_diff_desired = 0.
      self.torque_r_filtered = 0.
      self.torque_l_filtered = 0.
      self.turn_pid.reset()
      self.wheeled_speed_pid.reset()

    can_sends = []
    can_sends.append(bodycan.create_control(self.packer, torque_l, torque_r))

    new_actuators = CC.actuators.as_builder()
    new_actuators.accel = torque_l
    new_actuators.torque = torque_r
    new_actuators.torqueOutputCan = torque_r

    self.frame += 1
    return new_actuators, can_sends
