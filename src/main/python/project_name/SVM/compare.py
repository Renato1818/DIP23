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
    def __init__(self, sift_comparer, terminal=False, display=True):
        self.sift_comparer = sift_comparer
        self.terminal = terminal
        self.display = display

    def compare_vector(self, database: db, image_test_vector):
        # Initialize an empty list to store results for each image
        all_results = []

        for image_test in image_test_vector:
            # Compare the current image with images from the database
            results = self.compare(database, image_test)
            all_results.append(results)
        
    def compare (self, database: db, image_test):  
        # Read image paths from the database
        types, database_image_paths = database.read_images()
        if not database_image_paths: 
            return -1

        # Compare the test image with images from the database
        results = self.compare_with_database(image_test, database_image_paths)

        ## Statistics ##        
        # Scope for each type of flowers
        types_with_scores = self.find_scope(types, results)        
        print(f"Best scope: {types_with_scores[0,0]} {types_with_scores[0,2]/types_with_scores[0,1]}")

        '''# The best type
        most_similar_type = self.best_scope(types_with_scores)
        print(most_similar_type)'''
        
        # Display the most similar image
        most_similar_result = max(results, key=lambda x: x.similarity_score)
        print(f"Most similar image: {most_similar_result.folder_name}")
        print(f"Similarity score: {most_similar_result.similarity_score}")

        '''
        # Display the most similar image
        most_similar_result = results[0]  # Assuming results is not empty
        for result in results:
            if result.similarity_score > most_similar_result.similarity_score:
                most_similar_result = result
                print(most_similar_result.similarity_score)


        print(f"Number of analyzed images: {len(results)}")
        print(f"Most similar folder: {most_similar_result.folder_name}")
        '''
        
        most_similar_img = cv2.imread(most_similar_result.database_path, cv2.IMREAD_COLOR)
        # Plot the images
        if self.display:
            self.plot_images(image_test, most_similar_img, most_similar_result)

        '''fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(cv2.imread(new_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        axs[0].set_title("Image to classify")
        axs[0].axis("off")
        axs[1].imshow(cv2.cvtColor(most_similar_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Most Similar Image\nType: {most_similar_result.folder_name}\nScore: {most_similar_result.similarity_score}")
        axs[1].axis("off")
        plt.show()'''

        sift.compare_images_sift(image_test, most_similar_result.database_path)
        
        return results

    
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
            result = ResultStructure(
                image_name=os.path.basename(new_image_path),
                folder_name=folder_name,
                similarity_score=similarity_score,
                database_path=database_image_path
            )
            results.append(result)
            
            if self.terminal:
                print(f"Image: {result.image_name}, Folder: {result.folder_name}, Score: {result.similarity_score}")
            
        return results
            
    def find_scope(self, types, results: list[ResultStructure]):
        types_array = np.array(types)
        types_with_scores = np.zeros((len(types), 3), dtype=object)

        for result in results:
            indices = np.where(types_array == result.folder_name)
            for i in indices[0]:
                types_with_scores[i, 0] = result.folder_name
                types_with_scores[i, 1] += 1                        #number images
                types_with_scores[i, 2] += result.similarity_score  #total scope
        
        # Filter out scores with 0 images
        types_with_scores = types_with_scores[types_with_scores[:, 1] > 0]
        
        # Print a bar plot
        if self.display:
            self.find_scope_plot_bar(types_with_scores)
            
        # Order types_with_scores by average similarity score in descending order
        order = np.argsort(types_with_scores[:, 2] / np.maximum(1, types_with_scores[:, 1]))[::-1]
        types_with_scores = types_with_scores[order]
        

        #return types_with_scores.tolist()
        return types_with_scores
    
    def find_scope_plot_bar(self, types_with_scores):
        type_names = types_with_scores[:, 0]
        num_images = types_with_scores[:, 1]
        avg_similarity = types_with_scores[:, 2] / np.maximum(1, num_images)  # Avoid division by zero

        '''fig, ax = plt.subplots(figsize=(10, 6))
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

        plt.show()'''
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ind = np.arange(len(type_names))

        bars = ax.bar(ind, avg_similarity, label='Average Similarity', color='tab:blue')

        ax.set_xlabel('Flower Types')
        ax.set_ylabel('Average Similarity')
        ax.set_title('Average Similarity by Flower Type')
        ax.set_xticks(ind)
        ax.set_xticklabels(type_names, rotation='vertical')  # Rotate labels vertically
        ax.legend()

        # Annotate bars with their corresponding values
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.02, f'{avg_similarity[i]:.2f}', ha='center')
        
        plt.show()
    
    def plot_images(self, new_image_path, most_similar_img, most_similar_result):
        # Plot the images
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(cv2.imread(new_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        axs[0].set_title("Image to classify")
        axs[0].axis("off")
        axs[1].imshow(cv2.cvtColor(most_similar_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Most Similar Image\nType: {most_similar_result.folder_name}\nScore: {most_similar_result.similarity_score}")
        axs[1].axis("off")
        plt.show()
    
