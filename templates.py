import os

import cv2
import numpy as np

from utils import norm
from utils import regionprobs


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

        bw2, contours, hierarchy = cv2.findContours(templates[template][3], 
                                                    cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_NONE)
        # sometimes we might encounter small areas that are not part of the mask so we delete them.
        for count, cnt in enumerate(contours):
            if len(cnt) < 20:
                del contours[count]

        # saves the contour data
        cnt = contours[0]
        sub_data['cnt'] = cnt

        x,y,w,h = cv2.boundingRect(cnt)
        
        # aspect ratio for further use        
        aspect_ratio = float(w)/h
        sub_data['ar'] = aspect_ratio

        # checks the average color of the object
        mean_val = cv2.mean(templates[template][4],mask = templates[template][1])
        sub_data['mv'] = mean_val
        
        # Centroid, (Major Axis, minor axis), orientation
        # saves the angle for further use
        (x, y), (maxa, mina), angle = cv2.fitEllipse(cnt)
        sub_data['angle'] = angle
        
        # adds all data of the contour to the dataset
        data[template] = sub_data
    
    return data