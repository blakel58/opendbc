import unittest

from opendbc.car import DT_CTRL, structs
from opendbc.car.body.carcontroller import MAX_ACCEL, MAX_DECEL, MAX_SPEED_INTEGRATOR, TORQUE_DEADBAND, CarController, rate_limit
from opendbc.car.body.interface import CarInterface
from opendbc.car.body.values import CAR, SPEED_FROM_RPM


def get_control(axis_speed: float = 0., axis_turn: float = 0., enabled: bool = True):
  CC = structs.CarControl()
  CC.enabled = enabled
  # joystickd sends accel = 4 * axis, torque = axis
  CC.actuators.accel = 4.0 * axis_speed
  CC.actuators.torque = axis_turn
  return CC.as_reader()


class TestBodyCarController(unittest.TestCase):
  def setUp(self):
    self.CI = CarInterface(CarInterface.get_non_essential_params(CAR.COMMA_BODY))

  def _step(self, CC, rpm_l: float = 0., rpm_r: float = 0.) -> tuple[float, float]:
    self.CI.CS.out.wheelSpeeds.fl = rpm_l
    self.CI.CS.out.wheelSpeeds.fr = rpm_r
    actuators, _ = self.CI.apply(CC, 0)
    return actuators.accel, actuators.torque  # torque_l, torque_r

  def test_rate_limit_ramps(self):
    # full speed (0.8 m/s) is reached in 0.8s at MAX_ACCEL, not in one step
    v, t = 0., 0.
    while v < 0.8:
      v = rate_limit(0.8, v, MAX_ACCEL, MAX_DECEL)
      t += DT_CTRL
    self.assertAlmostEqual(t, 0.8 / MAX_ACCEL, delta=2 * DT_CTRL)

    # stopping is quicker, and reversing slows down through zero first
    v, t = 0.8, 0.
    while v > 0.:
      v = rate_limit(-0.8, v, MAX_ACCEL, MAX_DECEL)
      t += DT_CTRL
    self.assertAlmostEqual(t, 0.8 / MAX_DECEL, delta=2 * DT_CTRL)

  def test_deadband_is_continuous(self):
    f = CarController.deadband_filter
    self.assertEqual(f(0., TORQUE_DEADBAND), 0.)
    self.assertLess(abs(f(0.1, TORQUE_DEADBAND) - f(-0.1, TORQUE_DEADBAND)), 1.)
    self.assertEqual(f(100., TORQUE_DEADBAND), 100. + TORQUE_DEADBAND)
    self.assertEqual(f(-100., TORQUE_DEADBAND), -100. - TORQUE_DEADBAND)

  def test_step_command_does_not_lurch(self):
    # full stick from rest: torque builds gradually instead of jumping to ~100
    CC = get_control(axis_speed=1.)
    torques = [self._step(CC)[0] for _ in range(10)]
    self.assertLess(abs(torques[-1]), 30)
    for prev, cur in zip(torques, torques[1:], strict=False):
      self.assertLess(abs(cur - prev), 10)

  def test_no_chatter_at_rest(self):
    # zero command with a noisy wheel speed reading: torque stays small instead of flipping +/- deadband
    CC = get_control()
    for i in range(50):
      rpm = 1. if i % 2 else -1.
      torque_l, torque_r = self._step(CC, rpm, rpm)
      self.assertLess(abs(torque_l), TORQUE_DEADBAND / 2)
      self.assertLess(abs(torque_r), TORQUE_DEADBAND / 2)

  def _spin(self, steps: int, friction: float, viscous: float):
    """A crude body on training wheels turning in place: each wheel needs `friction` torque to move at all."""
    CC = get_control(axis_turn=0.75)
    rpm_l = rpm_r = 0.
    for _ in range(steps):
      torque_l, torque_r = self._step(CC, rpm_l, rpm_r)
      for name, torque in (("l", torque_l), ("r", torque_r)):
        rpm = rpm_l if name == "l" else rpm_r
        drive = max(0., abs(torque) - friction) * (1 if torque > 0 else -1)
        rpm += (drive / viscous - rpm) * 0.05   # settles toward the speed the leftover torque can hold
        if name == "l":
          rpm_l = rpm
        else:
          rpm_r = rpm
    return SPEED_FROM_RPM * (rpm_l - rpm_r), self.CI.CC.speed_diff_desired

  def test_turns_as_much_as_asked_despite_friction(self):
    # friction like the recorded drive: the proportional term alone stalled at about half the asked turn
    diff, desired = self._spin(400, friction=25., viscous=0.35)
    self.assertAlmostEqual(abs(desired), 0.375)
    self.assertGreater(diff / desired, 0.85)
    self.assertLess(diff / desired, 1.15)   # and doesn't overshoot

  def test_turn_feedforward_has_the_right_sign(self):
    # with the wheels already at the asked difference there's no error, yet torque is still applied to hold it
    CC = get_control(axis_turn=1.)
    for _ in range(100):
      desired = self.CI.CC.speed_diff_desired
      torque_l, torque_r = self._step(CC, desired / SPEED_FROM_RPM / 2, -desired / SPEED_FROM_RPM / 2)
    self.assertLess(torque_l, 0)       # positive turn axis asks for left slower than right
    self.assertGreater(torque_r, 0)

  def _drive(self, axis_speed: float, seconds: float, blocked_until: float = 0., gain: float = 0.032, lag: float = 0.15):
    """A crude body driving straight: speed settles toward what the torque can hold, unless something blocks the wheels.

    The gain is from a recorded drive (about 20 torque held 0.32 m/s once past the 10 of friction).
    Returns (time, speed, torque) samples.
    """
    CC = get_control(axis_speed=axis_speed)
    speed, samples = 0., []
    for i in range(int(seconds / DT_CTRL)):
      t = i * DT_CTRL
      rpm = speed / SPEED_FROM_RPM
      torque_l, torque_r = self._step(CC, rpm, rpm)
      torque = (torque_l + torque_r) / 2.
      if t < blocked_until:
        speed = 0.
      else:
        drive = max(0., abs(torque) - TORQUE_DEADBAND) * (1 if torque > 0 else -1)
        speed += (drive * gain - speed) * DT_CTRL / lag
      samples.append((t, speed, torque))
    return samples

  def test_pushes_through_a_stall(self):
    # wheels blocked while asked for 0.3 m/s: torque keeps building instead of sitting at a few dozen
    samples = self._drive(0.375, 3., blocked_until=3.)
    torque_at = {round(t, 2): torque for t, _, torque in samples}
    self.assertGreater(torque_at[1.0], 80)
    self.assertGreater(torque_at[2.99], 200)
    # but the stored push is limited
    self.assertLessEqual(abs(self.CI.CC.wheeled_speed_pid.i), MAX_SPEED_INTEGRATOR + 1.)

  def test_does_not_leap_after_the_stall(self):
    # blocked for three seconds, then free: it may surge, but briefly, and then settles at the asked speed
    samples = self._drive(0.375, 8., blocked_until=3.)
    after = [(t, speed) for t, speed, _ in samples if t >= 3.]
    self.assertLess(max(speed for _, speed in after), 1.5)
    too_fast = [t for t, speed in after if speed > 0.6]
    self.assertLess(len(too_fast) * DT_CTRL, 0.6)
    for _, speed in after[-100:]:
      self.assertAlmostEqual(speed, 0.3, delta=0.03)

  def test_drives_steadily_without_a_stall(self):
    samples = self._drive(0.375, 4.)
    speeds = [speed for _, speed, _ in samples]
    self.assertLess(max(speeds), 0.34)   # no overshoot worth the name
    for speed in speeds[-100:]:
      self.assertAlmostEqual(speed, 0.3, delta=0.01)

  def test_disabled_resets(self):
    CC = get_control(axis_speed=1.)
    for _ in range(100):
      self._step(CC, 0.8 / SPEED_FROM_RPM / 2, 0.8 / SPEED_FROM_RPM / 2)
    self.assertEqual(self._step(get_control(enabled=False)), (0, 0))
    self.assertEqual(self.CI.CC.speed_desired, 0.)
    self.assertEqual(self.CI.CC.wheeled_speed_pid.i, 0.)


if __name__ == "__main__":
  unittest.main()
