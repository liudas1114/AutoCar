import math
import board
import time
import adafruit_pca9685
import busio
from adafruit_motor import motor
import cv2
import logging

from Object_detection import Obj
from Line_detection import Detect_Lane

import unittest

logger = logging.getLogger(__name__)

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (255, 0, 0)

class Car():
    SPEED = 0.9
    print("Created car")
    
    def __init__(self):
        print("__init__")
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = adafruit_pca9685.PCA9685(i2c)

        self.pca.frequency = 1500 #TOSTUDY
        print("Freq = 1500")

        self.pca.channels[4].duty_cycle = 0xFFFF
        self.motorFL = motor.DCMotor(self.pca.channels[5],self.pca.channels[6])
        self.pca.channels[9].duty_cycle = 0xFFFF
        self.motorFR = motor.DCMotor(self.pca.channels[7],self.pca.channels[8])
        self.pca.channels[10].duty_cycle = 0xFFFF
        self.motorRL = motor.DCMotor(self.pca.channels[11],self.pca.channels[12])
        self.pca.channels[15].duty_cycle = 0xFFFF
        self.motorRR = motor.DCMotor(self.pca.channels[13],self.pca.channels[14])
        self.led_channel = self.pca.channels[0]

        self.engine_smooth()
        self.led_on()

        self.traffic_sign_processor = Obj(self, self.SPEED)
        self.line_detector = Detect_Lane(self, self.SPEED)

        self.video = cv2.VideoCapture(0)
        if (self.video.isOpened() == False): 
            print("Error reading video file")
        
        frame_width = int(self.video.get(3))
        frame_height = int(self.video.get(4))
        self.FRAME_SIZE = (frame_width, frame_height)

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') #TOSDTUDY
        self.video_org = cv2.VideoWriter('vidorg1.avi', self.fourcc, 10, self.FRAME_SIZE)

    def led_on(self):
        self.led_channel.duty_cycle = 0xffff

    def led_off(self):
        self.led_channel.duty_cycle = 0

    def engine_smooth(self):
        print("engine_smooth")
        self.motorFL.decay_mode = (motor.SLOW_DECAY)
        self.motorFR.decay_mode = (motor.SLOW_DECAY)
        self.motorRL.decay_mode = (motor.SLOW_DECAY)
        self.motorRR.decay_mode = (motor.SLOW_DECAY)

    def test_motor(self):
        print("test_motor")
        self.motorFL.throttle = 1
        self.motorFR.throttle = 1
        self.motorRL.throttle = 1
        self.motorRR.throttle = 1
        time.sleep(4)
        self.motorFL.throttle = None
        self.motorFR.throttle = None
        self.motorRL.throttle = None
        self.motorRR.throttle = None

    def go_front(self, speed):
        self.motorFL.throttle = speed
        self.motorFR.throttle = speed
        self.motorRL.throttle = speed
        self.motorRR.throttle = speed
    
    def go_free(self):
        self.motorFL.throttle = 0
        self.motorFR.throttle = 0
        self.motorRL.throttle = 0
        self.motorRR.throttle = 0
    def go_exit(self):
        self.motorFL.throttle = None
        self.motorFR.throttle = None
        self.motorRL.throttle = None
        self.motorRR.throttle = None

    def __enter__(self):
        return self

    def __exit__(self, _type, value, traceback):
        print("__exit__")
        self.exit()

    def exit(self):
        self.go_free()
        self.go_exit()
        self.led_channel.duty_cycle = 0
        self.video.release()
        cv2.destroyAllWindows()
        self.led_off()

    def turn(self):

        # [0..180] -> [-1..1]

        if(self.curr_steering_angle < 90 and self.curr_steering_angle >=0):
        #     power = round(self.SPEED*2/90/9*self.curr_steering_angle/9, 2)
        #     power = 0.5
        #     print(power)
            self.motorFL.throttle = 0
            self.motorRL.throttle = 0
            self.motorFR.throttle = self.SPEED
            self.motorRR.throttle = self.SPEED
        elif (self.curr_steering_angle > 90 and self.curr_steering_angle <= 180):
        #     power = round(self.SPEED*2/90/9*(self.curr_steering_angle-90)/9, 2)
        #     power = 0.5
        #     print(power)
            self.motorFL.throttle = self.SPEED
            self.motorRL.throttle = self.SPEED
            self.motorFR.throttle = 0
            self.motorRR.throttle = 0
        elif (self.curr_steering_angle  == 90):
            self.motorFL.throttle = self.SPEED
            self.motorRL.throttle = self.SPEED
            self.motorFR.throttle = self.SPEED
            self.motorRR.throttle = self.SPEED
        print(1)
        

    def drive(self, speed = SPEED):
        # self.go_front(speed)
        while self.video.isOpened():
            # time.sleep(2)
            # cv2.waitKey(500) 
            ret, frame = self.video.read()
            self.h, self.w, _ = frame.shape
            self.curr_steering_angle = 90
  
            if ret == True: 

                
                frame_line = frame.copy()
                frame_obj = frame.copy()
                logging.debug('start OD')
                frame_obj  = self.traffic_sign_processor.process_objects_on_road(frame_obj)
                frame_both = frame_obj.copy()
                logging.debug('stop OD')
                k_thold = 160
                bottom_fraction_to_analyze = 0.5
                
                try:
                    frame_line = self.line_detector.k_filter_for_black(frame_line, k_thold)
                    labeled_img, cx_l, sl_l, cx_r, sl_r = self.line_detector.detect_lanes(frame_line, bottom_fraction_to_analyze,return_marked_image = True)
                except:
                    self.go_front(self.SPEED)
                    # cv2.waitKey(-1)
                    time.sleep(.3)
                    self.go_free()
                    logger.error('failed lane detection', exc_info=True)
                    continue

                lines = []
                if cx_l is not None:
                    self.line_detector._line_for_centerx_and_slope(frame_line, cx_l, sl_l, GREEN)
                    lines.append(self.line_detector.compute_line(cx_l, sl_l, self.h))
                if cx_r is not None:
                    self.line_detector._line_for_centerx_and_slope(frame_line, cx_r, sl_r, BLUE)
                    lines.append(self.line_detector.compute_line(cx_r, sl_r, self.h))
                logging.debug('stop LineD')
                lines = []
                if cx_l is not None:
                    self.line_detector._line_for_centerx_and_slope(frame_both, cx_l, sl_l, GREEN)
                    lines.append(self.line_detector.compute_line(cx_l, sl_l, self.h))
                if cx_r is not None:
                    self.line_detector._line_for_centerx_and_slope(frame_both, cx_r, sl_r, BLUE)
                    lines.append(self.line_detector.compute_line(cx_r, sl_r, self.h))
                logging.debug('stop LineD')
                new_steering_angle = self.line_detector.steer(self.curr_steering_angle, lines, self.w, self.h)
                
                
                labeled_img = self.line_detector.display_heading_line(labeled_img, self.curr_steering_angle)

                frame_both = self.line_detector.display_heading_line(frame_both, new_steering_angle)

                
                cv2.imshow('Original', frame)
                cv2.imshow('Both', frame_both)
                cv2.imshow('Line detection', labeled_img)
                cv2.imshow('Object detection', frame_obj)
                
                self.curr_steering_angle = new_steering_angle
                self.turn()
                time.sleep(0.3)
                # cv2.waitKey(100) 
                self.go_free()

                # cv2.imshow('detect',frame_obj)



                # Press S on keyboard 
                # to stop the process
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
        
            # Break the loop
            else:
                self.exit() 
                break

    
        