import numpy as np
import os
import cv2

class Database:
    def __init__(self, database_path):
        self.database_path = database_path

    def read_image_paths(self, folder_path):
        """
        Reads image paths from the database folder.
        Returns a list of image paths. (image path, folder name).
        """        
        image_paths = []
        folder_name = os.path.basename(folder_path)
        
        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' does not exist.")
            return []

        """image_paths = [os.path.join(self.database_path, f) for f in os.listdir(self.database_path)
                       if f.endswith(('.jpg', '.png'))]"""

        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path) and f.endswith(('.jpg', '.png')):
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

        aux=0
        for subfolder in subfolders:
            image_paths.extend(self.read_image_paths(subfolder))
            
            #Controll
            print(subfolder)
            aux = aux +1
            if (aux > 5):
                return image_paths

        return image_paths