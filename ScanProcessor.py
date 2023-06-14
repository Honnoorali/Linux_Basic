import cv2
import numpy    as np
from labels     import label_parser
from database   import recognition_database

output_folder_location=r"./output"

class ScanProcessor:

    def __init__(self):
        self.processed_count = 0
        self.recognition_db = recognition_database()
        self.label_parser = label_parser()
        self.recognition_db.load_database("test.db")


    def copy_into_larger_image(self, w, h, img):
        new_img=np.full([w,h,3],255 ,dtype=int) 
        img_w,img_h=img.shape    
        x_offset=(w-img_w) 
        y_offset=(h-img_h)     
        for i in range(img_w):
            for j in range(img_h):
                new_img[i+x_offset][j+y_offset]=img[i][j]
        return new_img


        
    def compare_images(self, fingerprint_img, test_img, filename):
        x1w,x1h = fingerprint_img.shape
        x2w,x2h = test_img.shape
        
        w=max(x1w,x2w)
        h=max(x1h,x2h)
        
        stretch_near = cv2.resize(fingerprint_img, (x2h, x2w), interpolation = cv2.INTER_NEAREST)
        
        alligned_golden_img = self.copy_into_larger_image(w, h, stretch_near)    
        alligned_test_img = self.copy_into_larger_image(w, h, test_img)
        
        new_img=cv2.bitwise_xor(alligned_golden_img,alligned_test_img)
        
        same_pixels = np.sum(new_img == 0)
        similarity = (100* (new_img.size - same_pixels)) / new_img.size 
        cv2.imwrite(filename, new_img)
        #similarity = ssim(alligned_golden_img, alligned_test_img, multichannel=True)
        return similarity


    
    def process_print_image(self, frame):
        output_image = None
        print("Processing Frame:", self.processed_count)
        test_labels, num_test_items, output_image = self.label_parser.create_labelled_dictionary_from_frame_using_connected_model(frame)

        index = 0
        for entry in test_labels:
            cv2.imwrite("./output/scan_" + str(self.processed_count) +"_debug_"+str(index)+".png", entry[1])
            found, db_entry = self.recognition_db.find_entry(entry[0])
            if found == 0:
                print(entry[0], "Not found in the database")
                #source_w, source_h = entry[1].shape
                #aspect_ratio = source_w / source_h
                #print(source_w , source_h, "Aspect = ", aspect_ratio)
                #similarly_sized = recognition_db.get_similar_sized_entries(aspect_ratio)
                #print(similarly_sized)
                #print("\n")
            else:        
                fingerprint_image_color = db_entry[1]
                fingerprint_image_color_array = np.reshape(fingerprint_image_color, [db_entry[1].shape[0],db_entry[1].shape[1]])
                
                test_image = entry[1]
                
                filename = output_folder_location+"/diff_"+str(index)+".png"
                similarity = self.compare_images(fingerprint_image_color_array, test_image, filename)
                print(entry[0], "Comparing: ", similarity)
            index = index + 1


        cv2.imshow('Processed', output_image)
        self.processed_count += 1


