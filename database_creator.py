from labels import label_parser
from database import recognition_database
import cv2

def relabel_entry(labels, index, new_label):
    modify_item = list(labels[index])
    modify_item[0]= new_label
    labels[index]= tuple(modify_item)
    
    
dump_test_messages = 0
use_fpga_fingerprint_image = 0

recognition_db = recognition_database()
label_parser = label_parser()

#recognition_db.load_database("test.db")

#build the fingerprint database
if use_fpga_fingerprint_image:
    fingerprint_labels, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image(r'.\input\Printing.png')
    #Hack to correctly interpret Z
    modify_item = list(fingerprint_labels[25])
    modify_item[0]='Z'
    fingerprint_labels[25]= tuple(modify_item)
else:
    fingerprint_labels, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image(r'./input/printed_fingerprint.png')
    #Hack to correctly interpret Z
    fingerprint_labels.remove(fingerprint_labels[25])
    num_fingerprint_items = num_fingerprint_items -1
    relabel_entry(fingerprint_labels, 22, "W")

for entry in fingerprint_labels:
    recognition_db.add_entry(entry, 0)

#fingerprint_labels_lower_case, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image(r'./input/print_lower_case.png')
fingerprint_labels_lower_case, num_fingerprint_items = label_parser.create_labelled_dictionary_from_image_using_connected_model(r'./input/print_lower_case_cropped.png')

#fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[10])

#index = 0
#for entry in fingerprint_labels_lower_case:
#    print("index =", index, entry[0] )
#    cv2.imwrite("./output/labelled_out_" + str(index) +".png", entry[1])
#    index += 1

relabel_entry(fingerprint_labels_lower_case, 1, "b")
relabel_entry(fingerprint_labels_lower_case, 4, "e")
relabel_entry(fingerprint_labels_lower_case, 6, "g")
relabel_entry(fingerprint_labels_lower_case, 8, "i")
relabel_entry(fingerprint_labels_lower_case, 9, "j")
relabel_entry(fingerprint_labels_lower_case, 10, "k")
relabel_entry(fingerprint_labels_lower_case, 11, "l")
relabel_entry(fingerprint_labels_lower_case, 12, "m")
relabel_entry(fingerprint_labels_lower_case, 13, "n")
relabel_entry(fingerprint_labels_lower_case, 15, "p")
relabel_entry(fingerprint_labels_lower_case, 16, "q")
relabel_entry(fingerprint_labels_lower_case, 18, "s")
relabel_entry(fingerprint_labels_lower_case, 19, "t")
relabel_entry(fingerprint_labels_lower_case, 20, "u")
relabel_entry(fingerprint_labels_lower_case, 21, "v")
relabel_entry(fingerprint_labels_lower_case, 22, "w")
relabel_entry(fingerprint_labels_lower_case, 23, "x")
relabel_entry(fingerprint_labels_lower_case, 24, "y")
relabel_entry(fingerprint_labels_lower_case, 25, "z")

# fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[7])
# fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[7])
# fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[12])
# #fingerprint_labels_lower_case.remove(fingerprint_labels_lower_case[24])

# relabel_entry(fingerprint_labels_lower_case, 6, "g")
# relabel_entry(fingerprint_labels_lower_case, 11, "l")
# relabel_entry(fingerprint_labels_lower_case, 16, "q")

index = 0
for entry in fingerprint_labels_lower_case:
    recognition_db.add_entry(entry, 0)
    index = index + 1


recognition_db.persist_database("test.db")
