import cv2
import numpy as np

class Sift:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    #Compares two images using the SIFT algorithm.
    def compare_images(self, img1, img2):  
        
        # Compute SIFT descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
       
        if des1 is None or des2 is None:
            return 0

        # Ensure descriptors have the same data type
        des1 = np.float32(des1)
        des2 = np.float32(des2)
        
        # BFMatcher with default params
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = [m for m, n in matches if hasattr(m, 'distance') and hasattr(n, 'distance') and m.distance < 0.75 * n.distance]

        # Compute a similarity score (for example, the number of good matches)
        similarity_score = len(good_matches)
        
        return similarity_score

    def process_img_test(self, img1_path):        
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
        
        # Compute SIFT descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        
        # Handle the case where SIFT fails to detect keypoints or compute descriptors
        if des1 is None:
            return 0,0

        # Ensure descriptors have the same data type
        des1 = np.float32(des1)        
        img1_sift=kp1,des1
        
        return img1_sift
    
    #Compares two images but img1 is already in sift descriptors
    def compare_images_opt(self, img1_sift, img2_path):
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)
        
        # Compute SIFT descriptors
        kp1, des1 = img1_sift
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0
        
        des2 = np.float32(des2)
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        # Compute a similarity score
        similarity_score = len(good)
        
        return similarity_score
    