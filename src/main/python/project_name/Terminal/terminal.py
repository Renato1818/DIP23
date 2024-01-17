import numpy as np
import cv2
import pandas as pd

class Terminal:
    def __init__(self):
        return
    
    def similar_image(self, types_with_scores, most_similar_result):
        print(f"Most Similar Type: {types_with_scores[0,0]} (Average={types_with_scores[0,2]/types_with_scores[0,1]})")
        print(f"Most Similar Image: {most_similar_result.image_name} (type={most_similar_result.folder_name})")
        print(f"Most Similar score: {most_similar_result.similarity_score}")
    
    
    def display_results_table(self, all_results, expected_results):
        data = {
            "Image Path": [result.image_name for result in all_results],
            "Expected Label": expected_results,
            "Predicted Label": [result.folder_name for result in all_results],
            "Similarity Score": [result.similarity_score for result in all_results],
            #"Error Percentage": [100 * (1 if label != result.folder_name else 0) for result, (image_path, label) in zip(all_results, expected_results)],
        }

        df = pd.DataFrame(data)
        print("Results Table:")
        print(df)
        
    def accuracy_results(self, all_results, expected_results):
        correct_predictions = sum(1 for result, expected in zip(all_results, expected_results) if result.folder_name == expected)
        total_predictions = len(all_results)

        print(f"Correct_predictions: {correct_predictions} (total= {total_predictions})")
        accuracy = correct_predictions / total_predictions

        print(f"Accuracy: {accuracy}")