import os
import pickle
from time import time

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import ImageGrab

from src.process import ImageProcess as ip
import object_detection.utils.visualization_utils as vis_util


class Judge:

    def __init__(self, file_path, df_path):
        """
        Constructor for Judge class

        :param detection:
        :param categories:
        """
        if os.path.exists(file_path) or os.path.exists(df_path):
            self.__files = file_path
        else:
            raise OSError("File not exist")

        self.__df = pd.DataFrame.from_csv(path=df_path, index_col=1)

    def evaluate_tensorflow(self, detection, categories):

        with detection.as_default():
            with tf.Session(graph=detection) as sess:
                # Process all the files in selected directory
                for file in os.listdir(self.__files):

                    index = int(file.split(".")[0])

                    with open(file) as f:
                        data = pickle.load(f)
                        image = data['image']
                        detections = data['detections']

                    expected = len(detections)
                    evaluated = [self.tensorflow_analyse(x, sess) for x in detections]
                    #new_row = {'correct_tf':int, 'incorrect_tf':int, 'tf_detections':int, 'tf_time':float}
                    #self.__df[index] = new_row

    def tensorflow_analyse(self, image, sess):
        
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

