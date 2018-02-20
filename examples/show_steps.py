import os

import cv2
import matplotlib.pyplot as plt


def steps(example=0):

    from src.process import ImageProcess as ip
    from src.templates import make_templates
    from src.paths import DataPath

    options = [
        'norm', 'gray', 'blurred1', 'opening1', 
        'threshold1', 'masked1', 'opening2', 'blurred2', 
        'threshold2'
    ]

    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=50)
    br = cv2.BRISK_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ct_models = make_templates('CT', fast, br)

    dp = DataPath()

    try:
        file = os.listdir(dp.examples)[example]
    except KeyError:
        print('Invalid example')

    for option in options:

        image = cv2.imread(os.path.join(dp.examples, file))

        process = ip(image, ct_models, fast, br, bf, 'draw', debug=option)
        
        if option == 'norm':
            test_image = cv2.cvtColor(process.test, cv2.COLOR_BGR2RGB)
            plt.imshow(test_image)
        else:
            plt.imshow(process.test, cmap='gray')

        plt.show()


if __name__ == '__main__':

    import sys

    sys.path.append('.')

    steps()
