# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:06:29 2022

@author: akulkarni2
"""
import cv2
import numpy  as np
from labeled_entries import labelled_entry

class recognition_database:
    database_folder = "./database"
    prefix_str = "labeldata"
    m_collected_labels = []
    
    def __init__(self):
        self.m_collected_labels.clear()

    def get_similar_sized_entries(self, aspect_ratio):
        return_entry_list = []
        for entry in self.m_collected_labels:
            if (aspect_ratio-.1 > entry[3]) and (entry[3] < aspect_ratio+.1 ):
                return_entry_list.append(entry)
                
        return return_entry_list
    
    def find_entry(self, label_char):
        return_entry = []
        return_found = 0
        for entry in self.m_collected_labels:
            if entry[0] == label_char:
                return_entry = entry
                return_found = 1
                break
        return return_found, return_entry 

    def add_entry(self, label, overwriteprevious_label):
        add_label = 0
        #TODO, remove the previous entry if it exists
        found, entry = self.find_entry(label[0])
        if (found == 1):
            if (overwriteprevious_label == 0):
                self.m_collected_labels.remove(entry)
                add_label = 1
        else:
            add_label = 1
            
        if add_label == 1:
            print("Adding entry", label[0])
            self.m_collected_labels.append(label)
        else:
            print("Entry present for :", label[0])
        
    def remove_entry(self, label_char):
        found, entry = self.find_entry(label_char)
        if (found == 1):
            self.m_collected_labels.remove(entry)
            print("removing entry for:", label_char)
    
    def load_database(self, configuration_filename):
        print("Loading configuration from :", configuration_filename)
        self.m_collected_labels.clear()
        with open(configuration_filename, "r") as file:
            data = file.readlines()
            for line in data:
                line = line.strip()
                items = line.split("|")
                image_data = cv2.imread(items[1], cv2.IMREAD_COLOR)                
                converted_data = image_data[:,:,:-2]
                image_data = np.reshape(converted_data, [image_data.shape[0], image_data.shape[1]])
                image_w, image_h = image_data.shape
                aspect_ratio = image_w/image_h
                new_entry=labelled_entry(items[0], image_data, items[1], aspect_ratio)
                self.m_collected_labels.append(new_entry)

        
    def persist_database(self, configuration_filename):
        print("Saving configuration to :", configuration_filename)
        index = 0
        file = open(configuration_filename, 'w')
        for entry in self.m_collected_labels:
            file_name = self.database_folder + "/" + self.prefix_str + "_" + str(index) +".png"
            cv2.imwrite(file_name, entry[1])
            file.write(str(entry[0]) + "|" + file_name + "\n")
            index = index + 1
        file.close()
        
