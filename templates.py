import os

import cv2
import numpy as np

from utils import norm
import regionprobs as rp


# initialises fastfeaturedetector to get initial points of interests
fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=50)

# initialises Binary Robust Invariant Scalable Keypoints for 
# keypoint descriptor analyze
br = cv2.BRISK_create()

# BruteForce matcher to compare matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def make_templates(path):
    
    templates = {}
    
    for x in os.listdir(path):
        
        # reads the original RGB template image
        original = cv2.imread(path + x, 1)
        # normalises the image
        model_img = norm.normalize_image(original)
        # makes a grayscale image
        model_gray = cv2.cvtColor(model_img,cv2.COLOR_BGR2GRAY)
        # makes a rough mask of the image
        model_mask = cv2.inRange(model_gray, 0, 100)
        # fills the mask with gray image
        masked = cv2.bitwise_and(model_gray, model_mask)
        # takes the outer edges of the mask with gradient morph
        kernel = np.ones((3,3),np.uint8)
        gradient = cv2.morphologyEx(model_mask, cv2.MORPH_GRADIENT, kernel)
        # makes a data set of template_name : [images] for further usage
        templates[x.split(".")[0]] = [masked, model_mask, model_gray, gradient, model_img, original]

    return templates


def analyze_templates(templates):

    data = {}
    for template in templates:
        sub_data = {}

        # saves the templates descriptors
        kp = fast.detect(templates[template][0], None)
        kp, des = br.compute(templates[template][0], kp)
        sub_data['des'] = des

        stats = rp.RegionProbs(templates[template][3],mode='outer_full')
        
        if len(stats.get_properties()) > 1:
            print('Error in the template')

        stats = stats.get_properties()[0]
        # saves the contour data
        sub_data['cnt'] = stats.cnt
        
        sub_data['ar'] = stats.aspect_ratio
        

        # checks the average color of the object
        mean_val = cv2.mean(templates[template][4],mask = templates[template][1])
        sub_data['mv'] = mean_val
        
        sub_data['angle'] = stats.orientation
        
        # adds all data of the contour to the dataset
        data[template] = sub_data
    
    return data