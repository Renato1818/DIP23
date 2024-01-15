import numpy as np
import cv2
import matplotlib.pyplot as plt
import SVM.sift as sift
import SVM.sufr as surf
import DataBase.database as db
from Compare import compare as cp
from SVM import sift
import tkinter as tk
from tkinter.filedialog import askdirectory
import sys
from tqdm import tqdm

# Example usage
image1_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-1.jpg"
image2_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-2.jpg"
test_figure_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-3.jpg"
test_figure_path2 = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/download.jpg"

#### ASK DATABASE ####
# Set the paths for the database
'''database_path = askdirectory()
#database_path = path_directory()
if not database_path:
    print("No directory selected. Exiting program.")
    sys.exit()'''
database_path="C:/Users/asus/GitHub_clones/DIP23/src/resources/data_base"

#### ASK IMAGE TEST ####
new_image_path = test_figure_path2

# Initialize classes
database = db.Database(database_path)
image_comparer = cp.Compare(sift.Sift())

database.change_k(20)
image_comparer.compare_and_display_results(database)
##################################################################

'''database_path= "C:/Users/asus/GitHub_clones/DIP23/src/resources/data_base"
new_image_path = test_figure_path2 #to ask user

# Initialize classes
database = db.Database(database_path)
image_comparer = cp.Compare(sift.Sift())

types, database_image_paths = database.read_all_k_images()
if image_comparer.compare(database, types, new_image_path, database_image_paths) == -1:     
    print("Error")
    sys.exit()'''
