import cv2
from matplotlib import pyplot as plt
import numpy as np

class Sift:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def compare_images(self, img1, img2):
        """
        Compares two images using the SIFT algorithm.
        Returns a similarity score based on the number of good matches.
        """
        # Compute SIFT descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        print(des1)
        print('ok')
        print('des2', des2, type(des2))
        if des1 is None or des2 is None:
        # Handle the case where SIFT fails to detect keypoints or compute descriptors
            return 0

        # Ensure descriptors have the same data type
        des1 = np.float32(des1)
        des2 = np.float32(des2)
        
        # BFMatcher with default params
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        #good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        good_matches = [m for m, n in matches if hasattr(m, 'distance') and hasattr(n, 'distance') and m.distance < 0.75 * n.distance]


        # Compute a similarity score (for example, the number of good matches)
        similarity_score = len(good_matches)
        
        return similarity_score

    def compare_images_extra(self, img1_path):        
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
        
        # Compute SIFT descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        '''imgResult = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
        plt.title('SIFT Algorithm for image 1')
        plt.imshow(imgResult)
        plt.show()'''
        # Handle the case where SIFT fails to detect keypoints or compute descriptors
        if des1 is None:
            return 0,0

        # Ensure descriptors have the same data type
        des1 = np.float32(des1)        
        img1_sift=kp1,des1
        
        return img1_sift
    
    def compare_images_opt(self, img1_sift, img2_path):
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)
        
        # Compute SIFT descriptors
        kp1, des1 = img1_sift
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        '''imgResult = cv2.drawKeypoints(img2, kp1, None, color=(0,255,0), flags=0)
        plt.title('SIFT Algorithm for image 1')
        plt.imshow(imgResult)
        plt.show()'''
        '''print(des1)
        print('ok')
        print('des2', des2, type(des2))'''
        if des1 is None or des2 is None:
        # Handle the case where SIFT fails to detect keypoints or compute descriptors
            return 0
        
        # Ensure descriptors have the same data type
        des2 = np.float32(des2)
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        '''FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        bf = cv2.FlannBasedMatcher(index_params,search_params)'''
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        '''
        # Apply ratio test
        #good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        good_matches = [m for m, n in matches if hasattr(m, 'distance') and hasattr(n, 'distance') and m.distance < 0.75 * n.distance]
        '''
        #print (self.get_similarity_from_desc(des1, des2))
        #similarity_score = self.get_similarity_from_desc(des1, des2)
        #print("I'm here")

        # Compute a similarity score (for example, the number of good matches)
        similarity_score = len(good)
        
        return similarity_score
    
    def compare_images_sift(image1, image2):
        # Load images
        img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Draw matches
        '''img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("SIFT Matches", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

    def calculateMatches(self, des1,des2):
        matches = self.bf.knnMatch(des1,des2,k=2)
        topResults1 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                topResults1.append([m])
                
        matches = self.bf.knnMatch(des2,des1,k=2)
        topResults2 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                topResults2.append([m])
        
        topResults = []
        for match1 in topResults1:
            match1QueryIndex = match1[0].queryIdx
            match1TrainIndex = match1[0].trainIdx

            for match2 in topResults2:
                match2QueryIndex = match2[0].queryIdx
                match2TrainIndex = match2[0].trainIdx

                if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                    topResults.append(match1)
        return topResults

    def get_similarity_from_desc(self, img1, img2, approach = 'sift'):
        if approach == 'sift':
            # BFMatcher with euclidean distance
            bf = cv2.BFMatcher()
        else:
            # BFMatcher with hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        matches = bf.knnMatch(img1,img2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return len(good) / len(matches)