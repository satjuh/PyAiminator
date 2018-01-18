import os
import pickle
from time import time

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageGrab

from src.process import ImageProcess as ip
import object_detection.utils.visualization_utils as vis_util


class Judge:

    def __init__(self, images_path, df_path):
        """
        Constructor for Judge class

        :param detection:
        :param categories:
        """
        if os.path.exists(images_path) or os.path.exists(df_path):
            self.__files = images_path
            self.__out = df_path
        else:
            raise OSError("File not exist")

        self.__df = pd.DataFrame.read_csv(path=df_path)
        self.__new_df = pd.DataFrame(columns = ('correct_tf', 'incorrect_tf', 'tf_detections', 'tf_time'))

    def evaluate_tensorflow(self, detection, categories):

        self.__detection = detection
        self.__categories = categories
        self.__accepted = ["person", "backpack"]

        with detection.as_default():
            with tf.Session(graph=detection) as sess:
                # Process all the files in selected directory
                for file in os.listdir(self.__files):
                    print(file)
                    index = int(file.split(".")[0])

                    with open(self.__files+file, 'rb') as f:
                        data = pickle.load(f)
                        image = data['image']
                        detections = data['detections']

                    tf_time = time()
                    evaluate_image = self.tensorflow_analyse(image, sess)
                    tf_time = time() - tf_time

                    expected = len(detections)
                    evaluated = sum([1 for x in detections if self.tensorflow_analyse(x, sess) == 1])

                    correct = evaluated
                    incorrect = expected - evaluated

                    new_row = {'correct_tf':correct, 'incorrect_tf':incorrect, 'tf_detections':evaluate_image, 'tf_time':tf_time}

                    self.__new_df.loc[index] = new_row

                # Once the process is done, merge the new data to old df
                self.__df = pd.concat([self.__df, self.__new_df], axis=1)
                out = self.__out.split('.')[0] + '_tf.csv'
                self.__df.to_csv(out)

    def tensorflow_analyse(self, image, sess):

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.__detection.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.__detection.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.__detection.get_tensor_by_name('detection_scores:0')
        classes = self.__detection.get_tensor_by_name('detection_classes:0')
        num_detections = self.__detection.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores).astype(np.float64)

        detections = [1 for x in zip(classes, scores) if x[1] > .4 and self.__categories[x[0]]['name'] in self.__accepted]

        return sum(detections)


    def evaluate_X(self):
        pass

