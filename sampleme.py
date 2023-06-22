

# import os
# os.environ['TESSDATA_PREFIX'] = 'C:/Program Files(x86)/Tesseract-OCR/tessdata'





# import cv2

# # Read the image
# img = cv2.imread('C:\\Users\\sbante\\Updated_scripts\\input\\test_accel_decel.png')

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to enhance contrast
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# # Find contours
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Loop through contours and filter out non-text regions
# text_regions = []
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     if w / h > 0.5 and w > 20 and h > 20:
#         text_regions.append((x, y, w, h))

# # Extract text from image
# for i, region in enumerate(text_regions):
#     x, y, w, h = region
#     text = img[y:y+h, x:x+w]
#     cv2.imwrite(f'text_{i}.jpg', text)

# -*- coding: utf-8 -*-

import cv2
import numpy  as np
import pytesseract as pytesseract

from labeled_entries import labelled_entry
from pytesseract import Output

from PIL import Image
  
from labels import label_parser

import os


#tesseract_location = r'd:\tesseract-ocr\tesseract.exe'
# tesseract_location ='C:\ProgramData\anaconda3\Lib\site-packages\tesseract'
# tesseract_location ='C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
#c:\users\sbante\anaconda3\lib\site-packages
# C:\Users\sbante\Downloads\tesseract-ocr-setup-3.02.02 (1)
dump_test_messages = 2

padding = 5

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def bottom(self):
        return self.y + self.h
    
    def right(self):
        return self.x + self.w
    
    def area(self):
        return self.w * self.h

    def union(self, b):
        posX = min(self.x, b.x)
        posY = min(self.y, b.y)
        
        return Rect(posX, posY, max(self.right(), b.right()) - posX, max(self.bottom(), b.bottom()) - posY)
    
    def intersection(self, b):
        posX = max(self.x, b.x)
        posY = max(self.y, b.y)
        
        candidate = Rect(posX, posY, min(self.right(), b.right()) - posX, min(self.bottom(), b.bottom()) - posY)
        if candidate.w > 0 and candidate.h > 0:
            #print("intersection", candidate.x, candidate.y, candidate.w, candidate.h)
            return candidate
        return Rect(0, 0, 0, 0)
    
    def intersects(self, b):
        posX = max(self.x, b.x)
        posY = max(self.y, b.y)
        
        candidate = Rect(posX, posY, min(self.right(), b.right()) - posX, min(self.bottom(), b.bottom()) - posY)
        if candidate.w > 0 and candidate.h > 0:
            #print("intersection", candidate.x, candidate.y, candidate.w, candidate.h)
            return True
        return False
    
    
    def ratio(self, b):
        return self.intersection(b).area() / self.union(b).area()
    
    def print_rect(self):
        print(self.x, self.y, self.w, self.h)    



class label_parser:
    
    def add_border_to_image(self, img):
        row, col = img.shape[:2]
        bottom = img[row-2:row, 0:col]#bottom pixels to check clor of image
        mean = cv2.mean(bottom)[0]
        
        bordersize = padding
        border = cv2.copyMakeBorder(
            img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        return border
    
    def xor_pixel(self, image, value, x, y):
        if not (image[x][y]):
            image[x][y] = value
            #image[x][y] = 100
        
    def xor_area(self, image, value, left, upper, right, lower):
        #print("Xoring" , left, upper, right, lower)
        for x in range(lower, upper):
            for y in range(left, right):
                self.xor_pixel(image, value, x, y)

    def draw_box_area(self, image, value, left, upper, right, lower):
        start_point = (left, upper) 
        end_point = (right, lower)
        color = (value, value, value)
        thickness = 1
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
    def run_test_code(self, image, labels):
        height,width = image.shape
        text = pytesseract.image_to_boxes(image).split("\n")
        #print("text = " , text, len(text))
        index = 0
        for i in text:
            if i:
                (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1]))            
                upper = height - upper
                lower = height - lower
                extracted_image = image[lower:upper, left:right]
                extracted_image = self.add_border_to_image(extracted_image)

                aspect_ratio = extracted_image.shape[0]/extracted_image.shape[1]
                new_entry=labelled_entry(str(i[0]), extracted_image, "", aspect_ratio)
                labels.append(new_entry)
                index = index + 1        
                
    def consolidate_text_areas(self, test_boxes, height):
        ret_list = []
        rect_list = []
        for box in test_boxes:                    
            if box != "" :
                (left, upper, right, lower) = list(map(int, box.split(" ")[1:-1]))            
                upper = height - upper
                lower = height - lower
                r = Rect(left, upper, right-left, upper-lower)
                rect_list.append(r)

        temp = len(rect_list)
        index = 0
        while index < len(rect_list):
            final_rect = rect_list[index]
            index2 = index + 1
            
            while index2 < len(rect_list):
                #print("comparing ", index,"   " , index2)
                #rect_list[index2].print_rect()                
                
                if rect_list[index].intersects(rect_list[index2]) :
                    final_rect = final_rect.union(rect_list[index2])
                    rect_list.pop(index2)
                    print(index , "intersects", index2)
                    index2 = index2 -1
                    #final_rect.print_rect()                 
                    #print()
                index2 += 1
                    
            ret_list.append(final_rect)
            index  += 1
                    
        # index = 0
        # for rect in ret_list:
        #     print (index)
        #     rect.print_rect()
        #     index += 1
            
        print(len(ret_list), temp)
            
        return ret_list
    
    def predict_likely_char(self, image, text):
        ret_likely_char = ""
        previous_area = 0
        width,height = image.shape
        area = width * height
        #print(area)
        for i in text:
            if i:
                (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1]))            
                current_width = right - left
                current_height = lower - upper
                current_area = current_width * current_height
                #print(i, current_area, current_width, current_height)
                if current_area > previous_area:
                    ret_likely_char = i[0]  
                    previous_area = current_area
        print("Likely:", ret_likely_char)                    
        return ret_likely_char
    
    def deskew(self,image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
       
      
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dilated = cv2.dilate(binary, kernel, iterations=5)
        # eroded = cv2.erode(dilated, kernel, iterations=5)
       
        # lines using Hough Line Transform
        lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
       
       
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        avg_angle = np.mean(angles)
        if (avg_angle>0):
            avg_angle=-(180-avg_angle)
        # Rotate the image to correct the skew
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), (avg_angle), 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), flags=cv2.INTER_CUBIC)
       
        return avg_angle, rotated_image

        
    def create_labelled_dictionary_from_image_using_connected_model(self, image_filename):
        returned_labels = []    
        # pytesseract.pytesseract.tesseract_cmd= tesseract_location;

        # read image 
        img=cv2.imread(image_filename, cv2.IMREAD_COLOR)    
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./output/test-original.png", img)    
        
        img = self.add_border_to_image(img)
        
        height,width = img.shape       

        cv2.imwrite("./output/test-original.png", img)
        
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        cv2.imwrite("./output/test-blurred.png", blurred)
        
        otsu_threshold, image_result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        
        thresh =  image_result 
       
        cv2.imwrite("./output/test_after_threshold.png", image_result)
        
        text = pytesseract.image_to_boxes(thresh, config= '--psm 6').split("\n")   
      
        output=cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numlabels,labels,stats,centroids)=output
        
        if dump_test_messages == 2:
            print("text = " , text, len(text))
            print("Stats = ", stats)

        output = np.zeros(img.shape, dtype="uint8")       
        height,width = img.shape
        image_area = height * width
        print("Area = ", image_area )
        
        index = 0
        for i in range(1, numlabels):
              # Area of the component
            area = stats[i, cv2.CC_STAT_AREA]
            
            if (area/image_area) > 0.8:
                print("Processing Index:", i, "   ", stats[i])
                print("%area  =" , 100* area/image_area)
                componentMask = (labels == i).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask)
                # target = pytesseract.image_to_string(output, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                # print(target)
                
                text = pytesseract.image_to_boxes(thresh, config= '--psm 6 --oem 3').split("\n")
                #print("     text = " , text, len(text), self.consolidate_text_areas(text, height))
                
                area_lists = self.consolidate_text_areas(text, height)
                
                index2 = 0
                for area in area_lists:
                    
                    extracted_image = thresh[area.y-area.h:area.y, area.x:area.x+area.w]
                    extracted_image = self.add_border_to_image(extracted_image)
                    
                    if np.sum(extracted_image == 0):    
                        text1 = pytesseract.image_to_boxes(extracted_image, config= '--psm 10').split("\n")
                        print(type(text1))
                        
                        likely_char = self.predict_likely_char(extracted_image, text1)
                        
                        print(index2, area.w, area.h, text1, likely_char)
                        cv2.imwrite("./output/extracted_zzz_" + str(index2) +".png", extracted_image)
                        self.draw_box_area(output, 100, area.x-1, area.y-area.h+1, area.x+area.w+1, area.y-1)
                        cv2.putText(output, str(index2) + "  " + likely_char, (area.x, area.y-area.h), cv2.FONT_HERSHEY_PLAIN, 2, 0, 2, cv2.LINE_AA)
                        index2 = index2 + 1
                    
                index = 0
                for i in text:                    
                    if i != "" :
                        (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1]))            
                        upper = height - upper
                        lower = height - lower
                        extracted_image = thresh[lower:upper, left:right]
                        #cv2.putText(output, str(index), (100,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2);
                        
                        text1 = pytesseract.image_to_boxes(extracted_image, config= '--psm 8').split("\n")                    
                        #self.draw_box_area(output, 100, left, upper, right, lower)                        
                        cv2.imwrite("./output/extracted" + str(index) +".png", extracted_image)
                        index += 1
                                
        cv2.putText(output,'Hello World!', (100,30), cv2.FONT_HERSHEY_PLAIN, 1, 100, 2, cv2.LINE_AA)        
        cv2.imwrite("./output/test_output.png", output)
        return returned_labels, len(returned_labels)

    

# import cv2
# import numpy  as np
# import pytesseract as pytesseract
# from labels import label_parser

# from labeled_entries import labelled_entry
# from pytesseract import Output
# dump_test_messages = 2
# padding=5

# label_parser = label_parser()
# #tesseract_location = r'd:\tesseract-ocr\tesseract.exe'
# # tesseract_location =r'C:\Program Files (x86)\Tesseract-OCR'
# tesseract_location ='C:\Program Files (x86)\Tesseract-OCR'




# def create_labelled_dictionary_from_image_using_connected_model(image_filename):
#     returned_labels = []    
#     pytesseract.tesseract_cmd= tesseract_location;

#     # read image 
#     img=cv2.imread(image_filename, cv2.IMREAD_COLOR)    
#     img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("./demo/test-original.png", img)    
    
#     img = label_parser.add_border_to_image(img)
    
#     height,width = img.shape       

#     cv2.imwrite("./demo/test-original.png", img)
    
#     blurred = cv2.GaussianBlur(img, (3, 3), 0)
#     #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

#     cv2.imwrite("./demo/test-blurred.png", blurred)
    
#     otsu_threshold, image_result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        
#     thresh =  image_result 
   
#     cv2.imwrite("./demo/test_after_threshold.png", image_result)
    
#     text = pytesseract.image_to_boxes(thresh, config= '--psm 6').split("\n")   
  
#     output=cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
#     (numlabels,labels,stats,centroids)=output
    
#     if dump_test_messages == 2:
#         print("text = " , text, len(text))
#         print("Stats = ", stats)

#     output = np.zeros(img.shape, dtype="uint8")       
#     height,width = img.shape
#     image_area = height * width
#     print("Area = ", image_area )
    
#     index = 0
#     for i in range(1, numlabels):
#           # Area of the component
#         area = stats[i, cv2.CC_STAT_AREA]
        
#         if (area/image_area) > 0.8:
#             print("Processing Index:", i, "   ", stats[i])
#             print("%area  =" , 100* area/image_area)
#             componentMask = (labels == i).astype("uint8") * 255
#             output = cv2.bitwise_or(output, componentMask)
#             # target = pytesseract.image_to_string(output, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
#             # print(target)
            
#             text = pytesseract.image_to_boxes(blurred, config= '--psm 6 --oem 3').split("\n")
#             boxes=pytesseract.image_to_boxes(blurred,output_type=pytesseract.Output.DICT)
#             # cv2.imwrite("./demo/ext.png",boxes)
#             print("length of he box",len(boxes))
#             for i in range(len(boxes['char'])):
#                 char=text[i]
#                 x_min=boxes['left'][i]
#                 y_min=blurred.shape[0]-boxes['top'][i]
#                 x_max=boxes['right'][i]
#                 y_max=blurred.shape[0]-boxes['bottom'][i]
                
#                 char_image=blurred[y_min:y_max,x_min:x_max]
#                 cv2.imwrite("./demo/ext" + str(i) +".png", char_image)
                
#             #print("     text = " , text, len(text), self.consolidate_text_areas(text, height))
            
#             area_lists = label_parser.consolidate_text_areas(text, height)
            
#             index2 = 0
#             for area in area_lists:
                
#                 extracted_image = thresh[area.y-area.h:area.y, area.x:area.x+area.w]
#                 extracted_image =label_parser.add_border_to_image(extracted_image)
                
#                 if np.sum(extracted_image == 0):    
#                     text1 = pytesseract.image_to_boxes(extracted_image, config= '--psm 10').split("\n")
#                     print(type(text1))
                    
#                     likely_char = label_parser.predict_likely_char(extracted_image, text1)
                    
#                     print(index2, area.w, area.h, text1, likely_char)
#                     # cv2.imwrite("./demo/extracted_zzz_" + str(index2) +".png", extracted_image)
#                     label_parser.draw_box_area(extracted_image, 100, area.x-1, area.y-area.h+1, area.x+area.w+1, area.y-1)
#                     cv2.imwrite("./demo/extracted_zzz_" + str(index2) +".png", extracted_image)
#                     # cv2.putText(output, str(index2) + "  " + likely_char, (area.x, area.y-area.h), cv2.FONT_HERSHEY_PLAIN, 2, 0, 2, cv2.LINE_AA)
#                     index2 = index2 + 1
                
#             index = 0
#             for i in text:                    
#                 if i != "" :
#                     (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1]))            
#                     upper = height - upper
#                     lower = height - lower
#                     extracted_image = thresh[lower:upper, left:right]
#                     #cv2.putText(output, str(index), (100,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2);
                    
#                     text1 = pytesseract.image_to_boxes(extracted_image, config= '--psm 8').split("\n")                    
#                     #self.draw_box_area(output, 100, left, upper, right, lower) 
#                     # label_parser.draw_box_area(extracted_image, 100, left, upper, right, lower)                      
#                     cv2.imwrite("./demo/extracted" + str(index) +".png", extracted_image)
#                     index += 1
                            
#     cv2.putText(output,'Hello World!', (100,30), cv2.FONT_HERSHEY_PLAIN, 1, 100, 2, cv2.LINE_AA)        
#     cv2.imwrite("./output/test_output.png", output)
#     return returned_labels, len(returned_labels)


# # create_labelled_dictionary_from_image_using_connected_model( r'.\input\test_accel_decel.png')


    def extract_characters(self,image_path):
        # image_path=r'C:\Users\sbante_adm\Updated_scripts\input\Printing.png'
        # image = cv2.imread(image_path)

        output_folder=r"./demo"
        returned_labels = []   
        image= cv2.imread(image_path)
        # image = label_parser.add_border_to_image(img)
        skew_angle, rotated_image =self.deskew(image)
        gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        cv2.imwrite("./demo/blurred_image"+".png",  blurred)
        # cv2.imwrite("./output/test-blurred.png", blurred)
        
        otsu_threshold, image_result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        
        thresh =  image_result
        # gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)

        _, binary_image = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #needs to be inverted bcz findcontour get perfect contours in binary image
        cv2.imwrite("./demo/bianry_image"+".png", binary_image)
        
        # cv2.imshow('img',binary_image)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("contourslen",len(contours))
        character_count = 0
        _, binary_image = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY_INV )
        image_paths=[]
        
        for contour in contours:
            # print("contour",contour)
            x, y, w, h = cv2.boundingRect(contour)

            # cv2.rectangle(binary_image, (x-2, y-2), (x + w+1, y + h+1), (0 ,0, 0), 1)

            character= binary_image[y-2:y + h+2, x-2:x + w+2]
            # print(character)
            # custom_oem_psm_config = r'--oem 3 --psm 6'
            # cv2.imwrite("./demo/Character_image"+str(character_count)+".png", character)
            character=np.array(character)

# Load the image using OpenCV
# image = cv2.imread(r'C:\Users\sbante_adm\Updated_scripts\input\skew5.png')

# Convert the NumPy array to a PIL image
            pil_image = Image.fromarray(character)

# Save the PIL image
            pil_image.save('./demo/Character_image' + str(character_count) + '.png')

            # character.save('./demo/Character_image"+str(character_count)+".png')
            # character_path="./demo/Character_image"+str(character_count)+".png"
            # blurred = cv2.GaussianBlur(character, (3, 3), 0)
            # text=pytesseract.image_to_string( character_path)
            # print("text is ",text)
            image_paths.append("./demo/Character_image"+str(character_count)+".png")

            
          
            # List of image paths to be merged
            # image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
           
            
            
            # gray = cv2.cvtColor(character, cv2.COLOR_BGR2RGB)
            # otsu_threshold, image_result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        
            # thresh =  image_result 


            # # cv2.imwrite("./output/test-blurred.png", blurred)
            
            # otsu_threshold, image_result = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        
            # thresh =  image_result 
            # text=pytesseract.image_to_boxes(gray,config= '--oem 3  -l eng --psm 4 outputbase digits').split("\n")   
            # # for b in label.splitlines():
            # #     b = b.split(' ')
            # #     print(b)
            # print("label for "+str(character_count)+" character",text)
            # # output_file = os.path.join(output_folder, f'character_{character_count}.png')
            # cv2.imwrite(output_file, character)
            # characters.append([text,x,y,x+w,y+h])
            # cv2.imwrite("./demo/Character_image"+str(character_count)+".png", character)
            
            # label_position = (x, y - 10) # Position above the bounding rectangle
            # cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            character_count += 1



    # List of character image paths
        # character_paths = ['char1.png', 'char2.png', 'char3.png']
        
        # Apply OCR using Pytesseract to extract text for each character image
        # for i, image_path in enumerate(image_paths):
        #     # Load the character image
        #     character_image = cv2.imread(image_path)
            
        #     # Convert the image to grayscale
        #     gray_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2GRAY)
            
        #     # Apply thresholding to create a binary image
        #     _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
            
        #     # Apply OCR using Pytesseract to extract text
        #     character_text = pytesseract.image_to_string(binary_image,config= '--oem 3  -l eng --psm 10 outputbase digits')
        #     # print(f"Character {i+1}: {character_text}")

        
         
        # Open all the images
        images = [Image.open(path) for path in image_paths]
        
        # Calculate the maximum width and height among all images
        max_width = max(image.width for image in images)
        max_height = max(image.height for image in images)
        
        # Create a new blank image with the size to accommodate all images
        merged_image = Image.new('RGB', size = ((max_width * len(images)+50,max_height+50)),color = (255, 255, 255))
        
        # Paste each image onto the merged image
        x_offset = 0
        for image in images:
            merged_image.paste(image, (x_offset+20, 20))
            x_offset += image.width
        
        # Save the merged image as a PNG file
        merged_image .save('./demo/merged_image.png')
        # img_string=Image.open("./demo/merged_image.png")
        # # merged_image_path='./demo/merged_image.png'
        # merge_image=cv2.imread(Image.fromarray(merged_image))
        # img=cv2.cvtColor( merge_image, cv2.COLOR_BGR2RGB)
        # otsu_threshold, image_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)        
        # thresh =  image_result 
        merged_image=np.array(merged_image)
        # height,width = merged_image.shape[:2]
        im = Image.open('./demo/merged_image.png')

    #image size
        # width = im.size[0]
        height = im.size[1]
        # config = '--psm 6'  # Specify page segmentation mode
# boxes = pytesseract.image_to_boxes(image, config=config)

        text=pytesseract.image_to_boxes( merged_image,config=r'--tessdata-dir "C:\Users\sbante_adm\Updated_scripts\Tesseract-OCR\tessdata"').split("\n") 
        print("text is\n",text)
        for i in text:
            print("i in text",i)
            if i:
                (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1])) 
                
                upper = height - upper
                lower = height - lower
                extracted_image = merged_image[lower:upper, left:right]
        #         # extracted_image = self.add_border_to_image(extracted_image)
            
        #         if dump_test_messages == 1:
        #             print("Shapes", img.shape, thresh.shape, extracted_image.shape, extracted_image.size)
        #             print("Label:" , str(i[0]), left, upper, right, lower)            

                if np.sum(extracted_image == 0):
                    aspect_ratio = extracted_image.shape[0]/extracted_image.shape[1]
                    new_entry=labelled_entry(str(i[0]), extracted_image, "", aspect_ratio)
                    returned_labels.append(new_entry)
                     
                    # label_parser.xor_area(merged_image, 255, left, upper, right, lower)
                    # label_parser.draw_box_area(merged_image, 100, left, upper, right, lower)

        #             #index = index + 1        
        # cv2.imwrite("./output/extracted_data.png", extracted_data)
        # cv2.imwrite("./output/test_boxes.png", thresh_copy)
       
        # # img=cv2.imread(merged_image, cv2.IMREAD_COLOR)    
        # img=cv2.cvtColor( merged_image, cv2.COLOR_BGR2GRAY)

        # # img = self.add_border_to_image(img)
        # # height,width = img.shape
        
        # # cv2.imwrite("./output/test-original.png", img)
        # blurred = cv2.GaussianBlur(img, (3, 3), 0)
        # #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        
        # otsu_threshold, image_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)        
        # thresh =  image_result 
        
        # # cv2.imwrite("./output/test0.png", image_result)
        
        # #text = pytesseract.image_to_boxes(thresh).split("\n")
        # text = pytesseract.image_to_boxes(img, config= '--psm 10').split("\n")    
        
        # print("text = " , text, len(text))
        
        
        image_with_rectangles_path = os.path.join(output_folder, 'image_with_rectangles.png')
        # labels=pytesseract.image_to_boxes(binary_image,config= '--psm 6  ')
        # print("length lables",len(labels))
        # print("labels",labels)
        # cv2.imwrite(image_with_rectangles_path, image)
        cv2.imwrite("./demo/image_with_rectangles.png",binary_image)
        # print(character)
        # print(characters)
        
        # return character_count, image_with_rectangles_path
        # print(characters)
        return returned_labels, len(returned_labels)



    # num_characters, image_with_rectangles_path = extract_characters(r'.\input\test_098765.png')

    # print(f"Total characters found: {num_characters}")
    # print(f"Image with rectangles saved at: {image_with_rectangles_path}")
    # extract_characters(r'.\input\test_098765.png')







