# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 20:49:57 2022

@author: Anup Kulkarni
"""

# Python program to explain cv2.putText() method
    
# importing cv2
import cv2
import numpy  as np
    
# path
path = r'd:\test_image.png'
    
# Reading an image in default mode
image = np.zeros((200,200), dtype="uint8")
print(image.shape)
    
# Window name in which image is displayed
window_name = 'Image'
  
# font
font = cv2.FONT_HERSHEY_SIMPLEX

  
# # org
org = (10, 10)
  
# # fontScale
fontScale = 1
   
# # Blue color in BGR
color = (255, 200, 100)
  
# # Line thickness of 2 px
thickness = 2
   
# # Using cv2.putText() method
image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
   
# # Displaying the image
cv2.imshow(window_name, image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 