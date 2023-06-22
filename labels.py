# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:06:25 2022

@author: va
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:11:30 2022

@author: akulkarni2
"""
import cv2
import numpy  as np
import pytesseract as pytesseract

from labeled_entries import labelled_entry
from pytesseract import Output


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
                
                text = pytesseract.image_to_boxes(thresh, config=r'--tessdata-dir "C:\Users\sbante_adm\Updated_scripts\Tesseract-OCR\tessdata"').split("\n")
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

    def create_labelled_dictionary_from_image(self, image_filename):
        returned_labels = []    
        # pytesseract.tesseract_cmd= tesseract_location;

        # read image 
        img=cv2.imread(image_filename, cv2.IMREAD_COLOR)    
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.add_border_to_image(img)
        height,width = img.shape
        
        cv2.imwrite("./output/test-original.png", img)
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        
        otsu_threshold, image_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)        
        thresh =  image_result 
        thresh_copy = image_result.copy()
        extracted_data = image_result.copy()
        cv2.imwrite("./output/test0.png", image_result)
        
        #text = pytesseract.image_to_boxes(thresh).split("\n")
        text = pytesseract.image_to_boxes(thresh,config=r'--tessdata-dir "C:\Users\sbante_adm\Updated_scripts\Tesseract-OCR\tessdata"').split("\n")    
        if dump_test_messages == 2:
            print("text = " , text, len(text))
        
        #text.clear()
        if  len(text) > 1:            
            print(len(text))   
            #index = 0
            for i in text:
                print("i in text",i)
                if i:
                    (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1]))            
                    upper = height - upper
                    lower = height - lower
                    extracted_image = thresh[lower:upper, left:right]
                    # extracted_image = self.add_border_to_image(extracted_image)
                
                    if dump_test_messages == 1:
                        print("Shapes", img.shape, thresh.shape, extracted_image.shape, extracted_image.size)
                        print("Label:" , str(i[0]), left, upper, right, lower)            

                    if np.sum(extracted_image == 0):
                        aspect_ratio = extracted_image.shape[0]/extracted_image.shape[1]
                        new_entry=labelled_entry(str(i[0]), extracted_image, "", aspect_ratio)
                        returned_labels.append(new_entry)
                         
                        self.xor_area(extracted_data, 255, left, upper, right, lower)
                        self.draw_box_area(thresh_copy, 100, left, upper, right, lower)

                        #index = index + 1        
            cv2.imwrite("./output/extracted_data.png", extracted_data)
            cv2.imwrite("./output/test_boxes.png", thresh_copy)
            # print(" returned_labels",returned_labels)  
        else:
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
                    
            dict=pytesseract.image_to_boxes(thresh, output_type=Output.DICT,config=r'--tessdata-dir "C:\Users\sbante_adm\Updated_scripts\Tesseract-OCR\tessdata"')    
            output=cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
            (numlabels,labels,stats,centroids)=output

            print(stats)
            i = 0
            for entry in stats:
                #cv2.imwrite("./output/statstest"+str(i)+".png", thresh)
                lower = entry[1]
                upper = lower + entry[3]
                left = entry[0]
                right = left + entry[2]
                test_extracted_image = thresh[lower:upper, left:right]
                self.run_test_code(test_extracted_image, returned_labels)
                
                if (len(returned_labels) > 2):
                    print("====")
                    
                self.xor_area(thresh, 255,  left, upper, right, lower)                
                print("Entry:", i, len(returned_labels), "   ", entry[0], entry[1], entry[2], entry[3])
                i = i+1
                
                
            # for i in range(len(dict["left"])):
            #     #print("Processing", dict["char"][i], dict["left"][i], dict["bottom"][i], dict["right"][i], dict["top"][i])
            #     labelled_char = dict["char"][i]
            #     left = dict["left"][i]
            #     upper = dict["bottom"][i]
            #     right = dict["right"][i]
            #     lower = dict["top"][i]                
            #     upper = height - upper
            #     lower = height - lower
                
            #     extracted_image = thresh[lower:upper, left:right]
            #     extracted_image = self.add_border_to_image(extracted_image)
            
            #     #print("Processing 2:", labelled_char, "   ", left, upper, right, lower)
            #     self.xor_area(thresh, 255, left, upper, right, lower)
                
            #     aspect_ratio = extracted_image.shape[0]/extracted_image.shape[1]
            #     new_entry=labelled_entry(labelled_char, extracted_image, "", aspect_ratio)
            #     returned_labels.append(new_entry)
            
            cv2.imwrite("./output/test1.png", thresh)
            #cv2.imwrite("./output/test2.png", blurred)

            # num_recognized_labels = len(dict['char'])
            # j = 0;
            # for i in range(1,numlabels):   
            #     x=stats[i,cv2.CC_STAT_LEFT]
            #     y=stats[i,cv2.CC_STAT_TOP]
            #     w=stats[i,cv2.CC_STAT_WIDTH]
            #     h=stats[i,cv2.CC_STAT_HEIGHT]
                
            #     extracted_image = img[y:y+h, x:x+w]
            #     extracted_image = self.add_border_to_image(extracted_image)
            #     label_char = " "
            #     if j < num_recognized_labels :
            #         label_char = dict['char'][j]
                    
            #     if label_char != " ":
            #         self.xor_area(thresh, 255, x, height -y, x+w, height - h)
            #         aspect_ratio = extracted_image.shape[0]/extracted_image.shape[1]
            #         new_entry=labelled_entry(label_char, extracted_image, "", aspect_ratio)
            #         returned_labels.append(new_entry)
            #     j = j + 1            
   
        return returned_labels, len(returned_labels)
       
