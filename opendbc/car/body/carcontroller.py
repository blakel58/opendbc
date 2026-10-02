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
# the integrators are what push through a stall (a door sill, a rug, a wheel that drags): when the
# body isn't moving as asked, torque keeps building until it does. measured before this: stalled at a
# small bump the speed integrator added 3.5 torque a second, so it never got near what the motors can do,
# and one wheel barely turned in reverse. the limits keep the stored push small enough that the body
# doesn't leap forward once it gets over
MAX_SPEED_INTEGRATOR = 180.  # torque
MAX_TURN_INTEGRATOR = 120.   # torque
INTEGRATOR_UNWIND = 3.       # the stored push is given back this much faster than it builds

# torque applied up front for the speed that's asked for, instead of waiting for an error to build.
# measured on training wheels: asked for a 0.38 m/s wheel speed difference it only reached 0.18,
# because the proportional term alone can't beat the friction of turning in place
SPEED_FEEDFORWARD = 20.   # torque per m/s
TURN_FEEDFORWARD = 60.    # torque per m/s of wheel speed difference

# limits on how fast the speed targets may change, so a step on the joystick
# becomes a ramp (a trapezoid instead of a box). slowing down is allowed to be quicker
MAX_ACCEL = 1.0        # m/s^2
MAX_DECEL = 2.5        # m/s^2
MAX_TURN_ACCEL = 2.0   # m/s^2, wheel speed difference
MAX_TURN_DECEL = 4.0   # m/s^2

# a wheel that is asked to turn but isn't moving gets extra torque, quickly, until it breaks free.
# measured turning on the spot: one wheel stuck for 1.4 s while its torque crept from 43 to 66,
# then let go and overshot. holding 0.2 m/s takes about 25, so a stuck wheel needs far more to
# start than to keep going, and the integrators (tuned for while it's moving) build too slowly for that
STALL_SPEED = 0.02        # m/s, a wheel slower than this isn't moving
STALL_MIN_DESIRED = 0.04  # m/s, and it only counts if it's asked for more than this
STALL_KICK_RATE = 200.    # torque per second added while it's stuck
STALL_KICK_MAX = 200.     # torque
STALL_KICK_DECAY = 0.15   # seconds for the extra torque to fade once the wheel moves

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
    self.turn_pid = PIDController(110, k_i=100., k_f=TURN_FEEDFORWARD, rate=1 / DT_CTRL)
    self.wheeled_speed_pid = PIDController(160, k_i=200., k_f=SPEED_FEEDFORWARD, rate=1 / DT_CTRL)

    self.speed_desired = 0.
    self.speed_diff_desired = 0.
    self.torque_r_filtered = 0.
    self.torque_l_filtered = 0.
    self.kick_l = 0.
    self.kick_r = 0.

  @staticmethod
  def stall_kick(kick: float, desired: float, measured: float) -> float:
    # desired and measured are one wheel's speed in m/s
    if abs(desired) > STALL_MIN_DESIRED and abs(measured) < STALL_SPEED:
      return float(np.clip(kick + np.sign(desired) * STALL_KICK_RATE * DT_CTRL, -STALL_KICK_MAX, STALL_KICK_MAX))
    return kick * (1. - DT_CTRL / STALL_KICK_DECAY)

  @staticmethod
  def deadband_filter(torque, deadband):
    # fade the compensation in around zero. a hard +/- deadband flips sign with
    # the noise when the torque is near zero, which chatters the motors at rest
    return torque + deadband * float(np.clip(torque / TORQUE_DEADBAND_BLEND, -1., 1.))

  @staticmethod
  def update_pid(pid, error, feedforward, max_integrator, stalled=False):
    # stop building once the integrator holds as much push as it's allowed. and while a wheel is stuck
    # the kick is doing the pushing: that fades the moment the wheel moves, where an integrator would
    # carry its push on into a lurch
    freeze = stalled or (abs(pid.i) >= max_integrator and error * pid.i > 0)
    # and give the push back quickly once the error turns around (it got over the bump)
    if error * pid.i < 0:
      pid.i += (INTEGRATOR_UNWIND - 1.) * error * pid.k_i * DT_CTRL
    return pid.update(error, feedforward=feedforward, freeze_integrator=freeze)

  def update(self, CC, CS, now_nanos):

    torque_l = 0
    torque_r = 0

    if CC.enabled:
      # Read these from the joystick
      # TODO: this isn't acceleration, okay?
      self.speed_desired = rate_limit(CC.actuators.accel / 5., self.speed_desired, MAX_ACCEL, MAX_DECEL)
      self.speed_diff_desired = rate_limit(-CC.actuators.torque / 2., self.speed_diff_desired, MAX_TURN_ACCEL, MAX_TURN_DECEL)

      # break a stuck wheel free. (left - right) is the measured difference, so each wheel's own
      # target is the common speed plus or minus half the desired difference
      self.kick_l = self.stall_kick(self.kick_l, self.speed_desired + self.speed_diff_desired / 2., SPEED_FROM_RPM * CS.out.wheelSpeeds.fl)
      self.kick_r = self.stall_kick(self.kick_r, self.speed_desired - self.speed_diff_desired / 2., SPEED_FROM_RPM * CS.out.wheelSpeeds.fr)
      stalled = abs(self.kick_l) > 1. or abs(self.kick_r) > 1.

      speed_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl + CS.out.wheelSpeeds.fr) / 2.
      speed_error = self.speed_desired - speed_measured

      torque = self.update_pid(self.wheeled_speed_pid, speed_error, self.speed_desired, MAX_SPEED_INTEGRATOR, stalled)

      speed_diff_measured = SPEED_FROM_RPM * (CS.out.wheelSpeeds.fl - CS.out.wheelSpeeds.fr)
      turn_error = speed_diff_measured - self.speed_diff_desired
      # the turn error is measured minus desired, so its feedforward has the opposite sign
      torque_diff = self.update_pid(self.turn_pid, turn_error, -self.speed_diff_desired, MAX_TURN_INTEGRATOR, stalled)

      # Combine 2 PIDs outputs
      torque_r = torque + torque_diff + self.kick_r
      torque_l = torque - torque_diff + self.kick_l

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
      self.kick_l = 0.
      self.kick_r = 0.
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
