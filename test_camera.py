import unittest
import time
import cv2

class Test_Camera(unittest.TestCase):
    def setUp(self):
        self.video = cv2.VideoCapture(0)
        if (self.video.isOpened() == False): 
            print("Error reading video file")
        frame_width = int(self.video.get(3))
        frame_height = int(self.video.get(4))
        self.size = (frame_width, frame_height)

    def test_camera_with_video_window(self):
        timeout = time.time() + 7
        
        while(time.time() < timeout):
            ret, frame = self.video.read()
            if ret == True: 
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)
            else:
                break
        
        self.video.release()
        cv2.destroyAllWindows()


    def test_camera_with_saving_video(self):

        result = cv2.VideoWriter('tested_videocheck.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, self.size)
        timeout = time.time() + 7
        while(time.time() < timeout):
            ret, frame = self.video.read()
        
            if ret == True: 
                result.write(frame)
            else:
                break
        
        self.video.release()
        result.release()
        cv2.destroyAllWindows()
    


if __name__ == '__main__':
    unittest.main()