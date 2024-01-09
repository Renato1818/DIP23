import numpy as np
import os
import cv2
import random

class Database:
    def __init__(self, database_path):
        self.database_path = database_path
        self.k = 100

    
    def read_image_paths_random(self, folder_path):
        """
        Reads image paths from the database folder.
        Returns a list of image paths. (image path, folder name).
        """        
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
    
    def read_image_paths_random(self, folder_path, folder_name):
        """
        Reads image paths from the database folder.
        Returns a list of image paths. (image path, folder name).
        """        
        image_paths = []
        #folder_name = os.path.basename(folder_path)
        
        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' does not exist.")
            return []
        
        images_in_folder = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

        # Choose k random images
        selected_images = random.sample(images_in_folder, min(self.k, len(images_in_folder)))

        for selected_image in selected_images:
            file_path = os.path.join(folder_path, selected_image)
            image_paths.append((file_path, folder_name))
        
        return image_paths

    def read_images_from_subfolders(self):
        """
        Reads image paths from each subfolder within the specified database folder.
        Returns a list of tuples containing (image path, folder name).
        """
        image_paths = []

        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' does not exist.")
            return image_paths

        subfolders = [f.path for f in os.scandir(self.database_path) if f.is_dir()]
        types = []
        aux=0
        for subfolder in subfolders:
            folder_name = os.path.basename(subfolder)
            image_paths.extend(self.read_image_paths_random(subfolder, folder_name))
            types.append(folder_name)
                        
            """#Controll
            print(subfolder)
            aux = aux +1
            if (aux > 20):
                return image_paths"""

        return types,image_paths