import cv2
import numpy as np


def roi(img, vertices):
    """
    Masks the image according to vertices and creates the sort of "Region of interest"

    :param img: image we fill the region with
    :param vertices: points that determine the region of interest
    :return: masked image 
    """
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,np.int32([vertices]),255)
    
    return cv2.bitwise_and(img,mask)
