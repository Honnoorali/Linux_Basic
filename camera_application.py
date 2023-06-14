# import the opencv library
import cv2
import numpy as np
from matplotlib import pyplot as plt

import VideoProcessor as VP
import ScanProcessor as SP

alpha = 1.5 # Contrast control
#beta = 12 # Brightness control
beta = 8 # Brightness control
minimum_brightness = 2.8
start_point = (500, 250)
end_point = (1130, 600)
thickness = 2
capture_box_color = (200,0,0)

def process_frame(frame):
    global alpha
    global beta
    global start_point
    global end_point

    crop_frame = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]

    # call convertScaleAbs function
    adjusted = cv2.convertScaleAbs(crop_frame, alpha=alpha, beta=beta)

    #color = ('b','g','r')
    #for i,col in enumerate(color):
        #histr = cv2.calcHist([adjusted],[i],None,[256],[0,256])
        #plt.plot(histr,color = col)
        #plt.xlim([0,256])
    #plt.show()
    #print(histr)

    sp.process_print_image(adjusted)
    return True

def display_frame(frame):    
    global alpha
    global beta
    global minimum_brightness
    global start_point
    global end_point
    global thickness
    global capture_box_color

    cv2.imshow('Web cam', frame)

    cols, rows, color = frame.shape
    brightness = np.sum(frame) / (255 * cols * rows)
    ratio = brightness / minimum_brightness

    if ratio >= 1:
        adjusted = frame
    else:
        alpha = .8 / ratio
        #alpha = 1 #/ ratio
        adjusted = cv2.convertScaleAbs(frame, alpha = alpha, beta = beta)

    print(ratio, alpha, beta)

    adjusted = cv2.rectangle(adjusted, start_point, end_point, capture_box_color, thickness)
    cv2.imshow('corrected', adjusted)
    return True


vp = VP.VideoProcessor(0)
sp = SP.ScanProcessor()

#vp.show_input_frame(True)
vp.set_process_keystroke_fn(' ', process_frame)
vp.start_processing_frames(display_frame)
cv2.destroyAllWindows()
