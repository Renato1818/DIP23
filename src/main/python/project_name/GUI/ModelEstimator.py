#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:00:28 2023

@author: alessandrooddone
"""

import cv2
import numpy as np
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


class ModelEstimator:
    def __init__(self):
        self.model = tf.keras.saving.load_model("best_model.h5") 
        self.flower_labels = {"Astilbe" : 0,
                     "Bellflower" : 1,
                     "Black Eyed Susan" : 2,
                     "Calendula" : 3,
                     "California Poppy" : 4,
                     "Carnation" : 5,
                     "Common Daisy" : 6,
                     "Coreopsis" : 7,
                     "Daffodil" : 8,
                     "Dandelion" : 9,
                     "Iris" : 10,
                     "Magnolia" : 11,
                     "Rose" : 12,
                     "Sunflower" : 13,
                     "Tulip" : 14,
                     "Water Lily" : 15
                    }
        
    def predict(self, image_path):
        numericalImage = cv2.imread(str(image_path))
        resizedImage = cv2.resize(numericalImage, (224, 224))
        prediction = self.model.predict(np.array([resizedImage, ]))
        most_probable = np.argmax(prediction)
        confidence = np.max(prediction)
        return (np.array([i for i in self.flower_labels.keys()])[most_probable], str(confidence))
    
    def saliency(self, image_path):
        numericalImage = cv2.imread(str(image_path))
        resizedImage = cv2.resize(numericalImage, (224, 224))
        img = np.array([resizedImage, ])
 
        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(img[0].astype('double'), self.model.predict,  
                                                 top_labels=3, hide_color=0, num_samples=1000)

        #Select the same class explained on the figures above.
        ind =  explanation.top_labels[0]

        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

        #Plot. The visualization makes more sense if a symmetrical colorbar is used.
        plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
        plt.colorbar()
        plt.savefig('heatmap.png')

        # tmp, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        # tmp, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
        # ax1.imshow(mark_boundaries(img[0], mask_1))
        # ax2.imshow(mark_boundaries(img[0], mask_2))
        # ax1.axis('off')
        # ax2.axis('off')

        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        # img_boundary2 = mark_boundaries(img[0], mask)
        # plt.imshow(img_boundary2)
        # plt.savefig('saliencymap.png')
                               

                               