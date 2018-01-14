import os
from time import time

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import ImageGrab

from process import ImageProcess as ip
import object_detection.utils.visualization_utils as vis_util


class Judge:

    def __init__(self, file_path):
        """
        Constructor for Judge class

        :param detection:
        :param categories:
        """
        if os.path.exists(file_path):
            self.__files = file_path
        else:
            raise OSError("File doesn not exist")

    def evaluate_tensorflow(self, detection, categories):

        with detection.as_default():
            with tf.Session(graph=detection) as sess:
                # Process all the files in selected directory
                for file in os.listdir(self.__files):

                    image = cv2.imread(file)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image, axis=0)
                    image_tensor = detection.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection.get_tensor_by_name('detection_scores:0')
                    classes = detection.get_tensor_by_name('detection_classes:0')
                    num_detections = detection.get_tensor_by_name('num_detections:0')

                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    #classes = np.squeeze(classes).astype(np.int32)
                    #print(categories[classes[0]]['name'])

                    #TODO do something with the information
                    #TODO save the results in to pandas dataframe

    def evaluate_X(self):
        pass

