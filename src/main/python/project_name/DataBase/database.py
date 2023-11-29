import numpy as np
import os
import cv2

class Database:
    def __init__(self, database_path):
        self.database_path = database_path

    def read_image_paths(self):
        """
        Reads image paths from the database folder.
        Returns a list of image paths.
        """
        if not os.path.exists(self.database_path):
            print(f"Database path '{self.database_path}' does not exist.")
            return []

        image_paths = [os.path.join(self.database_path, f) for f in os.listdir(self.database_path)
                       if f.endswith(('.jpg', '.png'))]

        return image_paths

