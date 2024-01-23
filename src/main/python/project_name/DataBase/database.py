
import os
'''import numpy as np
import cv2'''
import random

class Database:
    def __init__(self, database_path):
        self.database_path = database_path
        self.k = 100
        self.test = 1

    def change_k(self, k: int):
        self.k = k
    
    def change_test(self, test: int):
        self.test = test
    
    #read all images inside a path
    def read_image_paths(self, folder_path):
        image_paths = []
        folder_name = os.path.basename(folder_path)
        
        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' does not exist.")
            return []

        image_paths = [os.path.join(self.database_path, f) for f in os.listdir(self.database_path)
                       if f.endswith(('.jpg', '.png'))]

        #To read all images
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path) and f.endswith(('.jpg', '.png')):
                image_paths.append((file_path, folder_name))
                
        return image_paths
    
    #read k random images inside a path
    def read_image_paths_random(self, folder_path, folder_name):      
        image_paths = []
        types = []
        
        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' does not exist.")
            return []
        
        images_in_folder = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

        # Choose k random images
        selected_images = random.sample(images_in_folder, min(self.k, len(images_in_folder)))

        for selected_image in selected_images:
            file_path = os.path.join(folder_path, selected_image)
            image_paths.append((file_path, folder_name))
        
        types.append(folder_name)
        return types, image_paths

    #read k random images inside a path and subfolders
    def read_all_k_images(self):
        image_paths = []
        types = []

        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' does not exist.")
            return types, image_paths
        
        for folder_path, _, _ in os.walk(self.database_path):
            folder_name = os.path.basename(folder_path)
            aux_types, aux_image_paths = self.read_image_paths_random(folder_path, folder_name)
            types.extend(aux_types)
            image_paths.extend(aux_image_paths)

        return types, image_paths
    
    def read_test_trainning_images(self):
        labels = []
        test_image_paths = []
        test_labels = []
        training_image_paths = []

        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' does not exist.")
            return labels, test_image_paths, test_labels, training_image_paths   

        for folder_path, _, _ in os.walk(self.database_path):
            folder_name = os.path.basename(folder_path)
            aux_labels, aux_image_paths = self.read_image_paths_random(folder_path, folder_name)
            if aux_image_paths:
                labels.extend(aux_labels)
                aux=0
                for file_path, folder_name in aux_image_paths:
                    if aux < self.test:
                        test_image_paths.append((file_path, aux_labels))
                        test_labels.extend(aux_labels)
                    else:
                        training_image_paths.append((file_path, folder_name))
                    aux+=1
                
        return labels, test_image_paths, test_labels, training_image_paths