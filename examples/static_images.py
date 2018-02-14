import os

import cv2
import matplotlib.pyplot as plt


def samples(path):
    # initialises fastfeaturedetector to get initial points of interests
    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=50)

    # initialises Binary Robust Invariant Scalable Keypoints for
    # keypoint descriptor analyze
    br = cv2.BRISK_create()

    # BruteForce matcher to compare matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    from src.process import ImageProcess as ip
    from src.templates import make_templates

    # initialize the data of the templates
    ct_models = make_templates('images/templates/CT/', fast, br)

    for file in os.listdir(path):

        image = cv2.imread(path + file)

        process = ip(image, ct_models, fast, br, bf, 'draw')

        image = cv2.cvtColor(process.image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.show()


if __name__ == '__main__':

    import sys

    sys.path.append('.')

    samples()
