import numpy as np
import cv2
import matplotlib.pyplot as plt
import DataBase.database as db
import GUI.gui_plot as pt
from Compare import compare as cp
from Compare import sift
import sys

# Example usage
image1_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-1.jpg"
image2_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-2.jpg"
test_figure_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-3.jpg"
test_figure_path2 = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/download.jpg"

database_path="C:/Users/asus/GitHub_clones/DIP23/src/resources/data_base"
#database_path = pt.GuiPlot.path_directory()
if database_path == -1:
    sys.exit()

#### ASK IMAGE TEST ####
#new_image_path = test_figure_path2

# Initialize classes
database = db.Database(database_path)
image_comparer = cp.Compare(sift.Sift())

# Run the code
database.change_k(10)
database.change_test(1)
image_comparer.trainning(database)

