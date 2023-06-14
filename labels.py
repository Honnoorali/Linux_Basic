import cv2
import numpy  as np
import pytesseract

from labeled_entries import labelled_entry
from pytesseract import Output

tesseract_location = r'/home/anupak/.local/bin/pytesseract'
dump_test_messages = 3
dump_interim_files = False
padding=4

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
    def __init__(self):
        self.debug_folder = r"./output/debug/"
        self.capture_debug_images = False
    
    def add_border_to_image(self, img):
        row, col = img.shape[:2]
        bottom = img[row-2:row, 0:col]
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
        thickness = 2
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
        ret_box_lists = []
        ret_list = []
        rect_list = []
        r_empty = Rect(0,0,0,0)
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
                    #print(index , "intersects", index2)
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
            
        #print(len(ret_list), temp)
            
        return ret_list

    def create_labelled_dictionary_from_image(self, image_filename):
        returned_labels = []    
        #pytesseract.pytesseract.tesseract_cmd= tesseract_location;

        # read image 
        img=cv2.imread(image_filename, cv2.IMREAD_COLOR)    
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.add_border_to_image(img)
        height,width = img.shape
        
        if dump_interim_files:
            cv2.imwrite("./output/test-original.png", img)

        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        
        otsu_threshold, image_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)        
        thresh =  image_result 
        thresh_copy = image_result.copy()
        extracted_data = image_result.copy()
        
        if dump_interim_files:
            cv2.imwrite("./output/test0.png", image_result)
        
        #text = pytesseract.image_to_boxes(thresh).split("\n")
        #text = pytesseract.image_to_boxes(thresh, lang='eng', config= '--psm 6') #.split("\n")    
        text = pytesseract.image_to_boxes(img).split("\n")

        if dump_test_messages == 2:
            print("text = " , text, len(text))
        
        #text.clear()
        if  len(text) > 1:            
                
            #index = 0
            for i in text:
                #print(i)
                if i:
                    (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1]))            
                    upper = height - upper
                    lower = height - lower
                    extracted_image = thresh[lower:upper, left:right]
                    extracted_image = self.add_border_to_image(extracted_image)
                
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
            if dump_interim_files:
                cv2.imwrite("./output/extracted_data.png", extracted_data)
                cv2.imwrite("./output/test_boxes.png", thresh_copy)
        else:
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
                    
            dict=pytesseract.image_to_boxes(thresh, output_type=Output.DICT,config= '--psm 6')    
            output=cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
            (numlabels,labels,stats,centroids)=output

            #print(stats)
            i = 0
            for entry in stats:
                #cv2.imwrite("./output/statstest"+str(i)+".png", thresh)
                lower = entry[1]
                upper = lower + entry[3]
                left = entry[0]
                right = left + entry[2]
                test_extracted_image = thresh[lower:upper, left:right]
                self.run_test_code(test_extracted_image, returned_labels)
                
                if dump_test_messages == 2:
                    if (len(returned_labels) > 2):
                        print("====")
                    
                self.xor_area(thresh, 255,  left, upper, right, lower)                
                
                if dump_test_messages == 2:
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
            
            if dump_interim_files:
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

    def create_labelled_dictionary_from_image_using_connected_model(self, image_filename):
        returned_labels = []    
        #pytesseract.pytesseract.tesseract_cmd = tesseract_location;

        # read image 
        img=cv2.imread(image_filename, cv2.IMREAD_COLOR)    
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if dump_interim_files:
            cv2.imwrite("./output/test-original.png", img)    

        img = self.add_border_to_image(img)
        
        height,width = img.shape       

        if dump_interim_files:
            cv2.imwrite("./output/test-original.png", img)
        
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        if dump_interim_files:
            cv2.imwrite("./output/test-blurred.png", blurred)
            #cv2.imwrite("./output/test-thresh.jpg", thresh)
        
        test = cv2.medianBlur(img, 5)
        th3 = cv2.adaptiveThreshold(test,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        
        if dump_interim_files:
            cv2.imwrite("./output/test_after_adaptive_threshold.png", th3)

        otsu_threshold, image_result = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        
        thresh = image_result 
        
        if dump_interim_files:
            cv2.imwrite("./output/test_after_threshold.png", image_result)
        
        text = pytesseract.image_to_boxes(thresh, config= '--psm 6').split("\n")
        #print(text)
        output=cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

        (numlabels,labels,stats,centroids) = output
        
        if dump_test_messages == 2:
            print("text = " , text, len(text))
            print("Stats = ", stats)

        output = np.zeros(img.shape, dtype="uint8")       
        height,width = img.shape
        image_area = height * width
        #print("Area = ", image_area )
        
        index = 0
        for i in range(1, numlabels):
              # Area of the component
            area = stats[i, cv2.CC_STAT_AREA]

            if (area/image_area) > 0.8:
                #print("Processing Index:", i, "   ", stats[i])
                #print("%area  =" , 100* area/image_area)

                componentMask = (labels == i).astype("uint8") * 200
                output = cv2.bitwise_or(output, componentMask)
                               
                edges = cv2.Canny(output, threshold1=100, threshold2=200) 
                if dump_interim_files:
                    cv2.imwrite("./output/test_edges.png", edges)

                #text = pytesseract.image_to_boxes(output, config= '--psm 4 --oem 3').split("\n")
                text = pytesseract.image_to_boxes(edges).split("\n")
                #print("     text = " , text, len(text), self.consolidate_text_areas(text, height))
                
                area_lists = self.consolidate_text_areas(text, height)
                
                index2 = 0
                for area in area_lists:
                    self.draw_box_area(output, 100, area.x, area.y-area.h, area.x+area.w, area.y)
                    cv2.putText(output, str(index2), (area.x, area.y-area.h), cv2.FONT_HERSHEY_PLAIN, 2, 0, 2, cv2.LINE_AA)
                    
                    extracted_image = thresh[area.y-area.h:area.y, area.x:area.x+area.w]
                    extracted_image = self.add_border_to_image(extracted_image)
                    text1 = pytesseract.image_to_boxes(extracted_image, config= '--psm 8').split("\n")
                    #cv2.imwrite("./output/extracted_zzz_" + str(index2) +".png", extracted_image)

                    # Attempt to find the character which covers moost of the area
                    index = 0
                    labeled_char = ' '

                    (e_width, e_height) = extracted_image.shape
                    e_area = e_width * e_height
                    count_black = e_area - cv2.countNonZero(extracted_image)
                    #print (index2, count_black)

                    if  count_black> 10:
                        for i in text1:                    
                            if i != "" :
                                label_char = i[0]
                                (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1]))            
                                upper = height - upper
                                lower = height - lower
                                covered_area = (right-left) * (upper-lower)
                                if (covered_area/e_area) > .70 :
                                    labeled_char = label_char

                        if (labeled_char != None):
                            #print(index2, labeled_char)
                            aspect_ratio = e_width / e_height
                            new_entry=labelled_entry(labeled_char, extracted_image, "", aspect_ratio)
                            returned_labels.append(new_entry)
                        else:
                            print("Could not label for index:", index2)

                    index2 = index2 + 1

                         #cv2.putText(out                                            ut, str(index), (100,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2);
                        
                #         text1 = pytesseract.image_to_boxes(extracted_image, config= '--psm 8').split("\n")                    
                #         #self.draw_box_area(output, 100, left, upper, right, lower)                        
                #         cv2.imwrite("./output/extracted" + str(index) +".png", extracted_image)
                #         index += 1
                                
        cv2.putText(output,'With Boxes (post intersection)!', (100,30), cv2.FONT_HERSHEY_PLAIN, 1, 100, 2, cv2.LINE_AA)        
        if dump_interim_files:
            cv2.imwrite("./output/test_output.png", output)
        return returned_labels, len(returned_labels)


    # This is a new function with clean implemntation for video related captures
    # Assumption is that the frame will be processed in the calling function
    def create_labelled_dictionary_from_frame_using_connected_model(self, image):       
        returned_labels = []    
        converted_image =  None

        image_height,image_width, image_color = image.shape
        image_area = image_height * image_width

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur( gray_img, (3, 3), 0)
        otsu_threshold, image_result = cv2.threshold(blurred_img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        

        if self.capture_debug_images:
            cv2.imwrite(self.debug_folder+"1_original_image.png", image)    
            cv2.imwrite(self.debug_folder+"2_gray_image.png", gray_img)
            cv2.imwrite(self.debug_folder+"3_blurred_image.png", blurred_img)
            cv2.imwrite(self.debug_folder+"4_threshold_image.png", image_result)

        converted_image = image_result.copy()

        output = cv2.connectedComponentsWithStats(image_result, 4, cv2.CV_32S)
        (numlabels, labels, stats, centroids) = output

        output = np.zeros(blurred_img.shape, dtype="uint8")       
        label_char = " "
        for i in range(1, numlabels):              
            component_area = stats[i, cv2.CC_STAT_AREA]             # Area of the component
            if (component_area/image_area) > 0.8:                   # Gate the content 
                componentMask = (labels == i).astype("uint8") * 200
                output = cv2.bitwise_or(output, componentMask)
                edges = cv2.Canny(output, threshold1=100, threshold2=200) 
                text = pytesseract.image_to_boxes(edges).split("\n")

                area_lists = self.consolidate_text_areas(text, image_height)

                index2 = 0

                for area in area_lists:
                    self.draw_box_area(converted_image, 100, area.x, area.y-area.h, area.x+area.w, area.y)
                    labeled_char = ' '

                    extracted_image = image_result[area.y-area.h:area.y, area.x:area.x+area.w]
                    extracted_image = self.add_border_to_image(extracted_image)

                    if self.capture_debug_images:
                        cv2.imwrite(self.debug_folder+"5_extracted_"+str(index2)+".png", extracted_image)

                        
                    (e_width, e_height) = extracted_image.shape
                    e_area = e_width * e_height
                    count_black = e_area - cv2.countNonZero(extracted_image)

                    if  count_black> 10:
                        text1 = pytesseract.image_to_boxes(extracted_image, config= '--psm 8').split("\n")

                        for i in text1:                    
                            if i != "" :
                                label_char = i[0]
                                (left, upper, right, lower) = list(map(int, i.split(" ")[1:-1]))            
                                upper = e_height - upper
                                lower = e_height - lower
                                covered_area = (right-left) * (upper-lower)
                                if (covered_area/e_area) > .70 :
                                    labeled_char = label_char

                        if (labeled_char != None):
                            aspect_ratio = e_width / e_height
                            new_entry = labelled_entry(labeled_char, extracted_image, "", aspect_ratio)
                            returned_labels.append(new_entry)
                        else:
                            print("Could not label for index:", index2)

                    cv2.putText(converted_image, str(label_char), (area.x, area.y-area.h), cv2.FONT_HERSHEY_PLAIN, 2, 0, 2, cv2.LINE_AA)
                    index2 = index2 + 1

        return returned_labels, len(returned_labels), converted_image
