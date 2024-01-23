import cv2
import numpy as np
from tqdm import tqdm
import numpy as np
import cv2
import DataBase.database as db
import Terminal.terminal as term
import GUI.gui_plot as pt
import os
'''
from Compare.sift import Sift
import pandas as pd
import tkinter as tk
from tkinter import ttk
from scipy.stats import mode'''


class ResultStructure:
    def __init__(self, image_name, folder_name, similarity_score, database_path):
        self.image_name = image_name
        self.folder_name = folder_name
        self.similarity_score = similarity_score
        self.database_path = database_path


class Compare:
    def __init__(self, sift_comparer, terminal=True, display=False):
        self.sift_comparer = sift_comparer
        self.terminal = terminal
        self.display = display
        self.plot = pt.GuiPlot()
        self.term = term.Terminal()
    
    #Receive the database and divide in two parts (Trainning and Test)
    #Later print the score
    def trainning(self, database: db):
        
        labels, test_image_paths, test_labels, database_image_paths = database.read_test_trainning_images()
        if not labels or not database_image_paths:
            print("No images to compare.")
            return

        all_results = self.input_vector_compare(labels, test_image_paths, database_image_paths)
        if all_results == -1:
            print("No results to compare.")
            return

        # Create a table to display the results
        self.term.display_results_table(all_results, test_labels)    
        self.term.accuracy_results(all_results, test_labels)  
            
    #Receive the database already divided
    #Vector of images to compare with the trainning part
    def input_vector_compare(self, types, test_image_paths, database_image_paths):
        
        all_results = []               
        for image_test, test_label in test_image_paths:
            if self.terminal:
                self.term.expected_label(test_label)
            # Compare the current image with images from the database
            results = self.compare(types, image_test, database_image_paths)
            
            if results != -1:
                all_results.append(results)

        if not all_results:
            return -1

        return all_results

    def compare(self, types, image_test, database_image_paths):
        if not database_image_paths:
            return -1

        results = self._sift_compare_test(image_test, database_image_paths)

        ## Statistics ##
        types_with_scores = self._find_scope(types, results)
        most_similar_result = max(results, key=lambda x: x.similarity_score)
        most_similar_img = cv2.imread(most_similar_result.database_path, cv2.IMREAD_COLOR)
        
        #Plot or print the results
        if self.terminal:
            self.term.similar_image(types_with_scores, most_similar_result)
        if self.display:
            self.plot.plot_images(image_test, most_similar_img, most_similar_result)

        return most_similar_result
    
    #To compare one image with the database
    def comparation(self, database: db, image_test):
        # Read image paths from the database
        types, database_image_paths = database.read_all_k_images()
        if not database_image_paths:
            return -1
        
        most_similar_result = self.compare(types, image_test, database_image_paths)
        return most_similar_result

    #Receive the test image and the database, and do the comparation
    def _sift_compare_test(self, new_image_path, database_image_paths):
        results = []    
        new_img = self.sift_comparer.process_img_test(new_image_path)

        for db_img_path, folder_name in tqdm(database_image_paths, desc="Comparing images", unit="image"):
            # Perform the comparison using the SIFT comparer
            similarity_score = self.sift_comparer.compare_images_opt(new_img, db_img_path)

            # Append the result to the list
            result = ResultStructure(
                image_name=os.path.basename(new_image_path),
                folder_name=folder_name,
                similarity_score=similarity_score,
                database_path=db_img_path,
            )
            results.append(result)

        return results

    def _find_scope(self, types, results: list[ResultStructure]):
        types_array = np.array(types)
        types_with_scores = np.zeros((len(types), 3), dtype=object)

        for result in results:
            indices = np.where(types_array == result.folder_name)
            for i in indices[0]:
                types_with_scores[i, 0] = result.folder_name
                types_with_scores[i, 1] += 1  # number images
                types_with_scores[i, 2] += result.similarity_score  # total scope

        # Filter out scores with 0 images
        types_with_scores = types_with_scores[types_with_scores[:, 1] > 0]

        # Print a bar plot
        if self.display:
            self.plot.find_scope_plot_bar(types_with_scores)
        
        # Order types_with_scores by average similarity score in descending order
        order = np.argsort(types_with_scores[:, 2] / np.maximum(1, types_with_scores[:, 1]))[::-1]
        types_with_scores = types_with_scores[order]

        return types_with_scores

