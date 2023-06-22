
from PIL import Image
  
from labels import label_parser
import cv2
import os
import pytesseract
import numpy  as np
import pytesseract as pytesseract

from labeled_entries import labelled_entry
from pytesseract import Output

tesseract_location ='C:\Program Files (x86)\Tesseract-OCR'
def extract_characters(image_path):
    output_folder=r"./demo"
    returned_labels = []   
    image= cv2.imread(image_path)
    # image = label_parser.add_border_to_image(img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
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
        cv2.imwrite("./demo/Character_image"+str(character_count)+".png", character)
        character_path="./demo/Character_image"+str(character_count)+".png"
        # blurred = cv2.GaussianBlur(character, (3, 3), 0)
        text=pytesseract.image_to_string( character_path)
        # print("text is ",text)
        image_paths.append("./demo/Character_image"+str(character_count)+".png")

        
      
        # List of image paths to be merged
        # image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
       
        
        
        gray = cv2.cvtColor(character, cv2.COLOR_BGR2RGB)
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
    text=pytesseract.image_to_boxes( merged_image,config='--psm 10').split("\n") 
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
# extract_characters(r'C:\Users\sbante_adm\Updated_scripts\input\Printing.png')
