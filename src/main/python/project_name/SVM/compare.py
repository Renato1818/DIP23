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
import os

class ResultStructure:
    def __init__(self, image_name, folder_name, similarity_score, database_path):
        self.image_name = image_name
        self.folder_name = folder_name
        self.similarity_score = similarity_score
        self.database_path = database_path


class Compare:
    def __init__(self, sift_comparer):
        self.sift_comparer = sift_comparer

    def compare (self, database: db, new_image_path):            

        # Read image paths from the database
        types, database_image_paths = database.read_images()
        if not database_image_paths: 
            return -1

        # Compare the test image with images from the database
        results = self.compare_with_database(new_image_path, database_image_paths)

        ## Statistics ##
        
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
        results = []

        # Load the new image
        new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

        for database_image_path, folder_name in tqdm(database_image_paths, desc="Comparing images", unit="image"):
            # Load the database image
            database_img = cv2.imread(database_image_path, cv2.IMREAD_GRAYSCALE)
            #print(f"Image: {database_image_path}; Is path: {folder_name}")

            # Perform the comparison using the SIFT comparer
            similarity_score = self.sift_comparer.compare_images(new_img, database_img)

            # Append the result to the list
            #results.append((os.path.basename(new_image_path), folder_name, similarity_score, database_image_path))
            result = ResultStructure(
                image_name=os.path.basename(new_image_path),
                folder_name=folder_name,
                similarity_score=similarity_score,
                database_path=database_image_path
            )
            results.append(result)

        return results
    
    '''def find_scope(self, types, results):        
        types_with_scores = [(t, 0, 0) for t in types]

        for database_image_path, folder_name, similarity_score in results:
            for i, (type_name, n_images, score) in enumerate(types_with_scores):
                if type_name == folder_name:
                    types_with_scores[i] = (type_name, n_images + 1, score + similarity_score)
        return types_with_scores'''
        
    def find_scope(self, types, results: ResultStructure):
        types_array = np.array(types)
        types_with_scores = np.zeros((len(types), 3), dtype=object)

        '''for image_name, folder_name, similarity_score, _ in results:
            indices = np.where(types_array == folder_name)
            for i in indices[0]:
                types_with_scores[i, 0] = folder_name
                types_with_scores[i, 1] += 1
                types_with_scores[i, 2] += similarity_score'''

        for result in results:
            indices = np.where(types_array == result.folder_name)
            for i in indices[0]:
                types_with_scores[i, 0] = result.folder_name
                types_with_scores[i, 1] += 1
                types_with_scores[i, 2] += result.similarity_score
        
        # Filter out scores with 0 images
        types_with_scores = types_with_scores[types_with_scores[:, 1] > 0]
        
        # Print a bar plot
        self.find_scope_plot_bar(types_with_scores)

        return types_with_scores.tolist()
    
    def find_scope_plot_bar(self, types_with_scores):
        type_names = types_with_scores[:, 0]
        num_images = types_with_scores[:, 1]
        avg_similarity = types_with_scores[:, 2] / np.maximum(1, num_images)  # Avoid division by zero

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        ind = np.arange(len(type_names))

        bars1 = ax.bar(ind, num_images, width, label='Number of Images')
        bars2 = ax.bar(ind + width, avg_similarity, width, label='Average Similarity')

        ax.set_xlabel('Flower Types')
        ax.set_ylabel('Count / Score')
        ax.set_title('Number of Images and Average Similarity by Flower Type')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(type_names, rotation='vertical')
        ax.legend()

        plt.show()
    
    def plot_bar(self, results):
        image_names = [result[0] for result in results]
        folder_names = [result[1] for result in results]
        similarity_scores = [result[2] for result in results]
        database_paths = [result[3] for result in results]

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        ind = np.arange(len(image_names))

        bars1 = ax.bar(ind, similarity_scores, width, label='Similarity Score')

        ax.set_xlabel('Image Names')
        ax.set_ylabel('Score')
        ax.set_title('Similarity Score by Image Name')
        ax.set_xticks(ind)
        ax.set_xticklabels(image_names, rotation='vertical')  # Rotate labels vertically
        ax.legend()

        plt.show()
    
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
