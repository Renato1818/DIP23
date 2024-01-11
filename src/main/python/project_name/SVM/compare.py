import cv2
import sqlite3
import numpy as np
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SVM.sift as sift
import SVM.sufr as surf
import DataBase.database as db
from SVM import compare as cp
from SVM import sift
from tqdm import tqdm

class Compare:
    def __init__(self, sift_comparer):
        self.sift_comparer = sift_comparer

    def compare (self, database, new_image_path):            

        # Read image paths from the database
        types, database_image_paths = database.read_images()
        if not database_image_paths: 
            return -1

        # Compare the new image with images from the database
        results = self.compare_with_database(new_image_path, database_image_paths)

        # Scope for each type of flowers
        types_with_scores = self.find_scope(types, results)        
        print(types_with_scores)

        # The best type
        most_similar_type = self.best_scope(types_with_scores)
        print(most_similar_type)


        # Display the most similar image
        i=0
        most_similar_path, most_similar_folder, most_similarity_score = results[i]
        for database_image_path, folder_name, similarity_score in results:
            if similarity_score > most_similarity_score:
                most_similar_path, most_similar_folder, most_similarity_score = results[i]
                print(most_similarity_score)
            i=i+1
                
        most_similar_img = cv2.imread(most_similar_path, cv2.IMREAD_COLOR)

        print(f"Number of analise images: {len(results)}" )
        print(f"Most similar folder: {most_similar_folder}" )

        # Plot the images 
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(cv2.imread(new_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        axs[0].set_title("Image to classify")
        axs[0].axis("off")
        axs[1].imshow(cv2.cvtColor(most_similar_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Most Similar Image\nType: {most_similar_folder}\nScore: {most_similarity_score}")
        axs[1].axis("off")
        plt.show()

        sift.compare_images_sift(new_image_path, most_similar_path)
    
    def compare_with_database(self, new_image_path, database_image_paths):
        """
        Compares the new image with each image in the database.
        Returns a list of similarity scores.
        """
        similarity_scores = []

        # Load the new image
        new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

        for database_image_path, folder_name in tqdm(database_image_paths, desc="Comparing images", unit="image"):
            # Load the database image
            database_img = cv2.imread(database_image_path, cv2.IMREAD_GRAYSCALE)
            #print(f"Image: {database_image_path}; Is path: {folder_name}")

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
