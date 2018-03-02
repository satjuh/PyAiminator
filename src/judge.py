import os
import pickle
import warnings
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageGrab

from src.paths import DataPath
from src.process import ImageProcess as ip

try:
    import tensorflow as tf
except ImportError:
    warnings.warn('Tensorflow not found')


class Judge:

    def __init__(self):
        """
        Constructor for Judge class
        """
        self.__dp  = DataPath()

    def evaluate_tensorflow(self, detection, categories):

        self.__detection = detection
        self.__categories = categories
        self.__accepted = ["person", "backpack"]

        df_columns = ('correct_tf', 'incorrect_tf', 'tf_detections', 'tf_time')
        name = 'tensorflow'

        with detection.as_default():
            with tf.Session(graph=detection) as sess:
                # Process all the files in directories
                for directory in os.listdir(self.__dp.collected):
                    result_file = os.path.join(self.__dp.dataframes, '{:s}_{:}.csv'.format(name, directory))
                    if not os.path.exists(result_file):
                        df = pd.DataFrame(columns = df_columns)
                        files = os.path.join(self.__dp.collected, directory)
                        for file in os.listdir(files):

                            index = int(file.split(".")[0])

                            with open(os.path.join(files, file), 'rb') as f:
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

                            new_row = {
                                'correct_tf':correct, 
                                'incorrect_tf':incorrect, 
                                'tf_detections':evaluate_image, 
                                'tf_time':tf_time
                            }

                            df.loc[index] = new_row

                        df.to_csv(result_file)

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

    def human_evaluation(self):

        for directory in os.listdir(self.__dp.collected):

            files = os.path.join(self.__dp.collected, directory)
            for file in os.listdir(files):

                print(file)

                with open(file, 'rb') as f:
                    data = pickle.load(file)
                    image = data['image']
                    detections = data['detections']

                #TODO show the image
                expected = input("How many humanoids is in the picture?(0-n): ")
                actual = sum([self.human_process(x) for x in detections])

    def human_process(self, image):

        #TODO show the image
        if input('Is this a humanoid?(Y/N)').capitalize() == 'Y':
            return 1
        else: 
            return 0
