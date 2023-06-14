import cv2
import numpy as np

class VideoProcessor:
    show_input_frame = False
    def __init__(self, file_name):
        self.name = file_name
        self.show_input_frame = False
        self.key_press_cb = None

    def set_process_keystroke_fn(self, keystroke, callback_function):
        self.key_press = keystroke
        self.key_press_cb = callback_function

    def show_input_frame(self, flag):
        self.show_input_frame = flag

    def start_processing_frames(self, process_func):
        self.vid = cv2.VideoCapture(self.name)

        if not self.vid.isOpened():
            print("Cannot open camera")
            exit()
            
        value = True
        while(value):
            self.ret, self.frame = self.vid.read()
            if self.show_input_frame:
                cv2.imshow('Input', self.frame)

            if self.ret:
                value = process_func(self.frame)
            
            key_pressed = cv2.waitKey(1)
            if key_pressed == ord('q'):
                break
            else:
                if ( key_pressed == ord(self.key_press) ):
                    value = self.key_press_cb(self.frame)

        self.vid.release()


    
