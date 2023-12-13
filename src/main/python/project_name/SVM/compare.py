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