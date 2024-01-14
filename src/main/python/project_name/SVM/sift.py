import cv2
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
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("SIFT Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

