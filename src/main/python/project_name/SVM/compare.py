import cv2
import sqlite3
import numpy as np

class Compare:
    def __init__(self, sift_comparer):
        self.sift_comparer = sift_comparer

    def compare_with_database(self, new_image_path, database_image_paths):
        """
        Compares the new image with each image in the database.
        Returns a list of similarity scores.
        """
        similarity_scores = []

        # Load the new image
        new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

        for database_image_path, folder_name in database_image_paths:
            # Load the database image
            database_img = cv2.imread(database_image_path, cv2.IMREAD_GRAYSCALE)
            print(f"Image: {database_image_path}; Is path: {folder_name}")

            # Perform the comparison using the SIFT comparer
            similarity_score = self.sift_comparer.compare_images(new_img, database_img)

            # Append the result to the list
            similarity_scores.append((database_image_path, folder_name, similarity_score))

        return similarity_scores
    
    def find_scope(self, types, results):        
        types_with_scores = [(t, 0, 0) for t in types]

        for database_image_path, folder_name, similarity_score in results:
            for i, (type_name, n_images, score) in enumerate(types_with_scores):
                if type_name == folder_name:
                    types_with_scores[i] = (type_name, n_images + 1, score + similarity_score)
        return types_with_scores
     
    def best_scope(self, types_with_scores):        
        k=0
        for type_name, n_images, score in types_with_scores:
            if k == 0:
                best_scope_name = type_name
                if (n_images != 0):
                    best_scope_type = score/n_images
                else:
                    best_scope_type = 0
                k=k+1                
            if (n_images != 0) and (score/n_images > best_scope_type):
                best_scope_type = score/n_images
                best_scope_name = type_name
        return best_scope_name
