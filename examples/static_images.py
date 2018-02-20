import os

import cv2
import matplotlib.pyplot as plt


def samples():

    from src.process import ImageProcess as ip
    from src.templates import make_templates
    from src.paths import DataPath

    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=50)
    br = cv2.BRISK_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ct_models = make_templates('CT', fast, br)

    dp = DataPath()

    for file in os.listdir(dp.examples):

        image = cv2.imread(os.path.join(dp.examples, file))

        process = ip(image, ct_models, fast, br, bf, 'draw')

        image = cv2.cvtColor(process.image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.show()


if __name__ == '__main__':

    import sys

    sys.path.append('.')

    samples()
