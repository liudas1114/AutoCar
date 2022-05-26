import unittest
import time

from Car import Car

class Test_Motor(unittest.TestCase):
    def test_motor_front_left_wheel_go_front(self):
        with Car() as car:
            car.motorFL.throttle = 1
            time.sleep(3)
            car.motorFL.throttle = 0
            car.motorFL.throttle = None
    def test_motor_front_right_wheel_go_front(self):
        with Car() as car:
            car.motorFR.throttle = 1
            time.sleep(3)
            car.motorFR.throttle = 0
            car.motorFR.throttle = None
    def test_motor_rear_left_wheel_go_front(self):
        with Car() as car:
            car.motorRL.throttle = 1
            time.sleep(3)
            car.motorRL.throttle = 0
            car.motorRL.throttle = None
    def test_motor_rear_right_wheel_go_front(self):
        with Car() as car:
            car.motorRR.throttle = 1
            time.sleep(3)
            car.motorRR.throttle = 0
            car.motorRR.throttle = None
    def test_motor_front_left_wheel_go_rear(self):
        with Car() as car:
            car.motorFL.throttle = -1
            time.sleep(3)
            car.motorFL.throttle = 0
            car.motorFL.throttle = None
    def test_motor_front_right_wheel_go_rear(self):
        with Car() as car:
            car.motorFR.throttle = -1
            time.sleep(3)
            car.motorFR.throttle = 0
            car.motorFR.throttle = None
    def test_motor_rear_left_wheel_go_rear(self):
        with Car() as car:
            car.motorRL.throttle = -1
            time.sleep(3)
            car.motorRL.throttle = 0
            car.motorRL.throttle = None
    def test_motor_rear_right_wheel_go_rear(self):
        with Car() as car:
            car.motorRR.throttle = -1
            time.sleep(3)
            car.motorRR.throttle = 0
            car.motorRR.throttle = None
    

class Test_LED(unittest.TestCase):
    def test_led_blinking(self):
        with Car() as car:
            time.sleep(1)
            car.led_off()
            time.sleep(1)
            car.led_on()
            time.sleep(2)
            car.led_off()
            time.sleep(1)
            car.led_on()
            time.sleep(3)

if __name__ == '__main__':
    unittest.main()