import cv2
import numpy as np


def normalize_image(img):
    """
    Normalizes the BGR image

    :param img: image that we want to normalize (needs to be bgr format)
    :return: normalized bgr image
    """
    # creates two identical empty pictures
    norm = np.zeros(img.shape, dtype=np.float32)
    norm_bgr = np.zeros(img.shape, dtype=np.uint8)

    # reads the b g r values from rgb picture
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    # counts the total value
    total = r + g + b

    # normalizes every value and sets values based on example: r' = r/total * 255.0
    np.seterr(divide='ignore', invalid='ignore')
    norm[:,:,0] = b/total * 255.0
    norm[:,:,1] = g/total * 255.0
    norm[:,:,2] = r/total * 255.0
    
    norm_bgr = cv2.convertScaleAbs(norm)

    return norm_bgr