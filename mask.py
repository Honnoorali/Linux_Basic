# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:42:06 2022

@author: va
"""

# now, let's make a circular mask with a radius of 100 pixels and
# apply the mask again

import numpy as np
import cv2
image=cv2.imread(r'.\input\Lowercase_abc.jpg')
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(mask, (145, 200), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
# show the output images
cv2.imwrite("Circular Mask.jpg", mask)
cv2.imwrite("Mask Applied to Image.jpg", masked)
cv2.imwrite("mask.jpg",masked)

#cv2.imshow('Masked Image',masked)