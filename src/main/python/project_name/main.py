import numpy as np
import cv2
import matplotlib.pyplot as plt
import SVM.sift as sift
import SVM.sufr as surf
import DataBase.database as db
from SVM import compare as cp
from SVM import sift
import tkinter as tk
from tkinter.filedialog import askdirectory
import sys

# Example usage
image1_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-1.jpg"
image2_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-2.jpg"

# Example usage
test_figure_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-3.jpg"
test_figure_path2 = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/download.jpg"

#sift.compare_images_sift(image1_path, image2_path)

#sift.compare_sum_of_images_with_external_figure(image1_path, image2_path, test_figure_path)

#surf.compare_images_surf(image1_path, image2_path)


# TEST# Set the paths for the database and the new image
database_path = askdirectory()
if not database_path:
    print("No directory selected. Exiting program.")
    sys.exit()
print(database_path)
#new_image_path = "path/to/new_image.jpg"
new_image_path = test_figure_path2

# Initialize classes
database = db.Database(database_path)
sift_comparer = sift.Sift()
image_comparer = cp.Compare(sift_comparer)

# Read image paths from the database
types, database_image_paths = database.read_images_from_subfolders()
if not database_image_paths: 
    print("No images found. Exiting program.")
    sys.exit()

# Compare the new image with images from the database
results = image_comparer.compare_with_database(new_image_path, database_image_paths)

# Print the results
"""aux=0
for database_image_path, similarity_score in results:
    print(f"Image: {aux}, Similarity Score: {similarity_score} \n")
    aux=aux+1"""

# Scope for each type of flowers
types_with_scores = image_comparer.find_scope(types, results)        
print(types_with_scores)

# The best type
most_similar_type = image_comparer.best_scope(types_with_scores)
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