import cv2
import numpy as np
import SVM.sift as sift
import SVM.sufr as surf

# Example usage
image1_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-1.jpg"
image2_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-2.jpg"

# Example usage
test_figure_path = "C:/Users/asus/GitHub_clones/DIP23/src/main/python/project_name/iris-3.jpg"

sift.compare_images_sift(image1_path, image2_path)


#sift.compare_sum_of_images_with_external_figure(image1_path, image2_path, test_figure_path)

#surf.compare_images_surf(image1_path, image2_path)