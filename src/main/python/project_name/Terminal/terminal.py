import numpy as np
import cv2

class Terminal:
    def __init__(self):
        return
    
    def similar_image(self, types_with_scores, most_similar_result):
        print(f"Most Similar Type: {types_with_scores[0,0]} (Average={types_with_scores[0,2]/types_with_scores[0,1]})")
        print(f"Most Similar Image: {most_similar_result.image_name} (type={most_similar_result.folder_name})")
        print(f"Most Similar score: {most_similar_result.similarity_score}")