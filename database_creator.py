 # -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:09:14 2022

@author: akulkarni2
"""
from labels import label_parser
from database import recognition_database
import cv2

def relabel_entry(labels, index, new_label):
    modify_item = list(labels[index])
    modify_item[0]= new_label
    labels[index]= tuple(modify_item)
    return labels 
    
dump_test_messages = 0
use_fpga_fingerprint_image = 0

recognition_db = recognition_database()
label_parser = label_parser()

#recognition_db.load_database("test.db")

#build the fingerprint database
if use_fpga_fingerprint_image:
    fingerprint_labels, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image(r'.\input\database image.png')
    #fingerprint_labels, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image(r'.\input\a.png')
    print("hi")
    # Hack to correctly interpret Z
    # modify_item = list(fingerprint_labels[25])
    # modify_item[0]='Z'
    # fingerprint_labels[25]= tuple(modify_item)
else:
    fingerprint_labels, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image(r'.\input\Printing.png')
    print("helo")
    #Hack to correctly interpret Z
    # fingerprint_labels.remove(fingerprint_labels[8])
    # num_fingerprint_items = num_fingerprint_items -1
    relabel_entry(fingerprint_labels, 0, "A")
    relabel_entry(fingerprint_labels, 22, "W")
    relabel_entry(fingerprint_labels, 24, "Y")
    
for entry in fingerprint_labels:
     recognition_db.add_entry(entry, 0)

fingerprint_labels_lower_case, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image_using_connected_model(r'.\input\printed_fingerprint.png')
#fingerprint_labels_lower_case, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image_using_connected_model(r'.\input\print_lower_case.png')
#fingerprint_labels_lower_case, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image_using_connected_model(r'.\input\a.png')

# fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[7])
# fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[7])
# fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[12])
# #fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[24])

# relabel_entry(fingerprint_labels_lower_case, 6, "g")
# relabel_entry(fingerprint_labels_lower_case, 11, "l")
# relabel_entry(fingerprint_labels_lower_case, 16, "q")

index = 0
for entry in fingerprint_labels_lower_case:
    file_name = "./output/" + str(index) +".png"
    #cv2.imwrite(file_name, entry[1])
    recognition_db.add_entry(entry, 0)
    index = index + 1

recognition_db.persist_database("test.db")
