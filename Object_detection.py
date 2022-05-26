import re
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class Obj():
    def load_labels(self, path='labels.txt'):
        """Loads the labels file. Supports files with or without index numbers."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    self.labels[int(pair[0])] = pair[1].strip()
                else:
                    self.labels[row_number] = pair[0].strip()

    def detect_objects(self, image):
        """Returns a list of detection results, each a dictionary of object info."""
        frame_obj = image
        img = cv2.resize(cv2.cvtColor(frame_obj, cv2.COLOR_BGR2RGB), (320,320))
            
        self.set_input_tensor(self.interpreter, img)
        self.interpreter.invoke()
        # Get all output details
        boxes = self.get_output_tensor(self.interpreter, 1)
        classes = self.get_output_tensor(self.interpreter, 3)
        scores = self.get_output_tensor(self.interpreter, 0)
        count = int(self.get_output_tensor(self.interpreter, 2))

        self.results = []
        print(self.results)
        for i in range(count):
            if scores[i] >= self.threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                self.results.append(result)

        img = cv2.resize(cv2.cvtColor(frame_obj, cv2.COLOR_BGR2RGB), (320,320))
        # res = self.detect_objects(interpreter, img, 0.2)
        # print(res)

        for result in self.results:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1,xmin * self.CAMERA_WIDTH))
            xmax = int(min(self.CAMERA_WIDTH, xmax * self.CAMERA_WIDTH))
            ymin = int(max(1, ymin * self.CAMERA_HEIGHT))
            ymax = int(min(self.CAMERA_HEIGHT, ymax * self.CAMERA_HEIGHT))
            
            cv2.rectangle(frame_obj,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            cv2.putText(frame_obj,self.labels[int(result['class_id'])],(xmin, min(ymax, self.CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 

        # cv2.imshow('Pi Feed', frame_obj)

        # if cv2.waitKey(10) & 0xFF ==ord('q'):
        #     cap.release()
        #     cv2.destroyAllWindows()
        
        return frame_obj

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        final_frame = self.detect_objects(frame)
        self.control_car(self.results)
        return final_frame
    
    def control_car(self, objects):
        car_state = {"speed": self.speed, "last_speed": self.speed}


        if(objects) == 0:
            print("No sign detected")
        contain_stop_sign = 0
        for obj in objects:
            print(obj)
            obj_label = self.labels[obj["class_id"]]
            process = self.sign[obj["class_id"]]
            if self.is_close_by(obj,self.CAMERA_HEIGHT):
                if(obj["class_id"] == 0):
                    self.car.go_front(self.speed)
                    print("brick")
                elif(obj["class_id"] == 1):
                    print("stop")
                    self.car.go_free()


    def is_close_by(self, obj, frame_height, min_height_pct=30/480): #TODO istrizaine
        ymin, xmin, ymax, xmax = obj['bounding_box']
        obj_height = ymax-ymin
        print(f'{obj_height} : {min_height_pct}')
        return obj_height > min_height_pct


    def __init__(self,
                car = None, 
                speed = 1,
                model='detect.tflite',
                label='labels.txt',
                threshold = 0.8,
                width = 640,
                height = 480):
        self.car = car
        self.speed = speed
        self.threshold = threshold
        self.CAMERA_WIDTH = width
        self.CAMERA_HEIGHT = height
        self.load_labels(label)

        self.interpreter = Interpreter(model)
        self.interpreter.allocate_tensors()
        _, input_height, input_width, _ = self.interpreter.get_input_details()[0]['shape']
        self.sign = {
            0: self.car.go_front(self.speed),
            1: self.car.go_free()}

    def set_input_tensor(self, interpreter, image):
        """Sets the input tensor."""
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)

    def get_output_tensor(self, interpreter, index):
        """Returns the output tensor at the given index."""
        output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
        return tensor

  