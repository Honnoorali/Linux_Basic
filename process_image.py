
import numpy  as np

import cv2
from skimage.metrics import structural_similarity as ssim

#import pytesseract
from labels import label_parser
from database import recognition_database
from PIL import Image, ImageDraw
  #2way
#from PIL import Image
import glob

import imutils

output_folder_location=r"./output"

def copy_into_larger_image(w,h,img):
    new_img=np.full([w,h,3],255,dtype=int) 
    img_w,img_h=img.shape    
    x_offset=(w-img_w) 
    y_offset=(h-img_h)     
    for i in range(img_w):
        for j in range(img_h):
            new_img[i+x_offset][j+y_offset]=img[i][j]
    return new_img


    
def compare_images(fingerprint_img, test_img, filename):
    x1w,x1h = fingerprint_img.shape
    x2w,x2h = test_img.shape
    
    w=max(x1w,x2w)
    h=max(x1h,x2h)
    
    stretch_near = cv2.resize(fingerprint_img, (x2h, x2w), interpolation = cv2.INTER_NEAREST)
    
    alligned_golden_img=copy_into_larger_image(w, h, stretch_near)    
    alligned_test_img=copy_into_larger_image(w, h, test_img)
    cv2.imwrite(" alligned_test_img.jpg",alligned_test_img)
    
    

    new_img=cv2.bitwise_xor(alligned_golden_img,alligned_test_img)

    cv2.imwrite("new_img.jpg",new_img)
    
    # np.sum(img==0) count the no pixels having the value "0"
    # img.size=total no of pixels in img
       
    same_pixels = np.sum(new_img == 0)
    similarity = (100* (new_img.size - same_pixels)) / new_img.size 
    dissimalirty = (100* np.sum(new_img != 0)) / new_img.size 
    print("XOR DIfference:" , dissimalirty)
  
    cv2.imwrite(filename, new_img)
    similarity = ssim(alligned_golden_img, alligned_test_img, multichannel=True)
    print("similarity of ssim  ",similarity)
    
    
   #1way
   
    diff=cv2.absdiff(alligned_golden_img,alligned_test_img)
    diff_score=np.sum(diff)/alligned_golden_img.shape[0]/alligned_test_img.shape[1]/3;
    print("diffrence_score ",diff_score)
    cv2.imwrite("diff.png ",diff)
    
    
    
    return similarity
 
# Main 
recognition_db = recognition_database()
label_parser = label_parser()

recognition_db.load_database("test.db")

#test_labels, num_test_items = label_parser.create_labelled_dictionary_from_image(r'.\input\test_accel_decel.png')
#test_labels, num_test_items = label_parser.create_labelled_dictionary_from_image(r'.\input\test_098765.png')
test_labels, num_test_items = label_parser.create_labelled_dictionary_from_image(r'C:\Users\sbante_adm\Updated_scripts\input\test_098765.png')

index = 0
for entry in test_labels:
    cv2.imwrite("./output/debug"+str(index)+".png", entry[1])    
    found, db_entry = recognition_db.find_entry(entry[0])
    if found == 0:
        
        print(entry[0], "Not found in the database")
        #source_w, source_h = entry[1].shape
        #aspect_ratio = source_w / source_h
        #print(source_w , source_h, "Aspect = ", aspect_ratio)
        #similarly_sized = recognition_db.get_similar_sized_entries(aspect_ratio)
        #print(similarly_sized)
        print("\n")
    else:        

        fingerprint_image_color = db_entry[1]
        fingerprint_image_color_array = np.reshape(fingerprint_image_color, [db_entry[1].shape[0],db_entry[1].shape[1]])
        
        test_image = entry[1]
     
        for entry in test_image:# i think it must be test_lables
            blur = cv2.GaussianBlur(entry, (3,3), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    
            cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 300:
                    cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
    
    
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
            # cv2.imwrite('thresh', thresh)
            # cv2.imwrite('close', close)
        
    
      
        # #removing unfound images
        # test_image = np.array(test_image)

        # test_image[321:56,359:60] = (255,255,255)
        # test_image = Image.fromarray(test_image)
        
        
        filename = output_folder_location+"/diff_"+str(index)+".png"
        similarity = compare_images(fingerprint_image_color_array, test_image, filename)
        print(entry[0], "Comparing: ", similarity)
    index = index + 1
print("  ")
   
def similarity_score():
    fingerprint_img=cv2.imread(r'.\input\test_accel_decel.png')
    test_img=cv2.imread(r'.\input\printed_fingerprint.png') 
    resize=fingerprint_img.shape[1],fingerprint_img.shape[0]
    test_img=cv2.resize(test_img,resize)
    TPC_fingerprint_img=np.sum(fingerprint_img)
    TPC_test_img=np.sum(test_img)
    print("TPC_fingerprint_img",TPC_fingerprint_img )
    print("TPC_fingerprint_img", TPC_test_img)  
    total_pixcel_count=TPC_fingerprint_img+TPC_test_img
    print("total pixcel",total_pixcel_count)
    print(" ")
    new_img=cv2.bitwise_xor(fingerprint_img,test_img)
    URPC=np.sum(new_img)
    print("unrecognised_pixcel",URPC) 
    RPC=TPC_test_img - URPC
    print("the RPC count",RPC)
    similarity = ssim(fingerprint_img, test_img, multichannel=True)
    final_score=((similarity*RPC)/100) +((URPC*similarity)/100)
    print(" ")
    print("final score  ",final_score)
    
similarity_score()
 

# def gloabal_score():
    
        
 
    
    
    
  