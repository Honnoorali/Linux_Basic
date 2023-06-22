import numpy  as np
import cv2
from skimage.metrics import structural_similarity as ssim
from labels import label_parser
from database import recognition_database
  #2way
from PIL import Image
import glob

import imutils

output_folder_location=r"./output"

def copy_into_larger_image(w,h,img):
    new_img=np.full([w,h,3],255 ,dtype=int) 
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
       
    same_pixels = np.sum(new_img == 0)
    similarity = (100* (new_img.size - same_pixels)) / new_img.size 
    dissimalirty = (100* np.sum(new_img != 0)) / new_img.size 
    print("XOR DIfference:" , dissimalirty)

    #similarity score 
    TPC_allignedgolden_img=np.sum(alligned_golden_img)
    TPC_test_img=np.sum(alligned_test_img)
    print("TPC_allignedgolden_img", TPC_allignedgolden_img)
    print("")
    total_pixcel_count=TPC_allignedgolden_img + TPC_allignedgolden_img
    print("total pixcel",total_pixcel_count)
    print(" ")
    URPC=np.sum(new_img)
    print("unrecognised_pixcel",URPC)
    print(" ")
    RPC=TPC_test_img - URPC
    print("the RPC count",RPC)
    
    
    """
    RPC=0
    for i in alligned_golden_img:
        for j in alligned_test_img:
            if j in i:
                RPC=RPC+1;
    print("the RPC are:",RPC) """
   
    cv2.imwrite(filename, new_img)
    similarity = ssim(alligned_golden_img, alligned_test_img, multichannel=True)
    print("similarity of ssim",similarity)
    
    
   #1way
    diff=cv2.absdiff(alligned_golden_img,alligned_test_img)
    diff_score=np.sum(diff)/alligned_golden_img.shape[0]/alligned_test_img.shape[1]/3;
    print("diffrence_score",diff_score)
    cv2.imwrite("diff.png",diff)   
    
    return similarity
        

# Main 
recognition_db = recognition_database()
label_parser = label_parser()

recognition_db.load_database("test.db")

test_labels, num_test_items = label_parser.create_labelled_dictionary_from_image(r'.\input\test_accel_decel.png')
#test_labels, num_test_items = label_parser.create_labelled_dictionary_from_image(r'.\input\test_098765.png')
test_labels, num_test_items = label_parser.create_labelled_dictionary_from_image(r'.\input\printed_fingerprint.png')

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
        
        filename = output_folder_location+"/diff_"+str(index)+".png"
        similarity = compare_images(fingerprint_image_color_array, test_image, filename)
        print(entry[0], "Comparing: ", similarity)
    index = index + 1
    

