from time import time

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import ImageGrab

from process import ImageProcess as ip
import object_detection.utils.visualization_utils as vis_util


class Judge:

    def __init__(self, detection, categories, templates, fast, br, bf):
        """
        Constructor for Judge class

        :param detection:
        :param categories:
        """
        self.__detection = detection
        self.__categories = categories
        self.__templates = templates
        self.__fast = fast
        self.__br = br
        self.__bf = bf

    def analyse_screen(self, width, heigth):

        with self.__detection.as_default():
            with tf.Session(graph=self.__detection) as sess:

                print("Starting session.\n")
                start = time()
                while True:

                    # captures the screen.
                    screen = np.array(ImageGrab.grab(bbox=(0,27,800,627)))
                    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

                    intensity, kp = self.process_frame(screen)

                    process_time = time()
                    process = ip(screen, self.__templates, self.__fast, self.__br, self.__bf)
                    process_time = time() - process_time

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(screen, axis=0)
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
                    # Visualization of the results of a detection.
                    classes = np.squeeze(classes).astype(np.int32)
                    print(self.__categories[classes[0]]['name'])

                    print("Loop took {:} seconds.".format(time()-start))

                    start = time()

                    cv2.imshow('AimAssistant', screen)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

    def analyse_video(self, path, frames):
        pass

    def process_frame(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        intensity = cv2.mean(gray)[0]
        kp = self.__fast.detect(gray, None)

        return intensity, kp
