import unittest

from opendbc.car import DT_CTRL, structs
from opendbc.car.body.carcontroller import MAX_ACCEL, MAX_DECEL, TORQUE_DEADBAND, CarController, rate_limit
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

  def test_disabled_resets(self):
    CC = get_control(axis_speed=1.)
    for _ in range(100):
      self._step(CC, 0.8 / SPEED_FROM_RPM / 2, 0.8 / SPEED_FROM_RPM / 2)
    self.assertEqual(self._step(get_control(enabled=False)), (0, 0))
    self.assertEqual(self.CI.CC.speed_desired, 0.)
    self.assertEqual(self.CI.CC.wheeled_speed_pid.i, 0.)


if __name__ == "__main__":
  unittest.main()
