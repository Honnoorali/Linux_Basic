# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:38:01 2023

@author: sbante
"""

import pytesseract
from PIL import Image

# Open image
#img = Image.convert('L')
tesseract_location =r'C:\Program Files (x86)\Tesseract-OCR'

img = Image.open('r.\input\test_098765.png')

# Perform OCR using TesseractC:\Users\sbante\Updated_scripts\input\test_accel_decel.png
gray_img = img.convert('L')
text = pytesseract.image_to_string(gray_img)

# Print extracted text
print(text)
