import sys

import cv2

from src import judge, process, templates, setup


def main():
    """
    Main function of the aiminator.py - captures the screen of the size 800x600 in
    the top-left corner of the screen and feeds a new window to display with detected
    models
    """
    # initialises fastfeaturedetector to get initial points of interests
    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=50)

    # initialises Binary Robust Invariant Scalable Keypoints for
    # keypoint descriptor analyze
    br = cv2.BRISK_create()

    # BruteForce matcher to compare matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # initialize the data of the templates
    ct_models = templates.make_templates('images/templates/CT/', fast, br)

    #dg, ci = setup.setup_tensor()
    data_path = "M:/Projects/PyAiminator"
    c = process.CollectProcess(ct_models, fast, br, bf, 'debug', path=data_path)
    c.collect_from_screen(800,600)
    #j = judge.Judge(data_path + "/data/data_0/", "df_0.csv")
    #j.evaluate_tensorflow(dg, ci)

if __name__ == '__main__':
    main()
