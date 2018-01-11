import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageGrab
from time import time

class Judge:

    def __init__(self, detection, category_index):
        
        self.__detection = detection
        self.__indexes = category_index

    def analyse_screen(self, width, heigth):
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                print("Starting session.\n")
                    start = time()
                    while True:

                        # captures the screen.
                        screen = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(0,27,800,627))), cv2.COLOR_RGB2BGR)
                        process = ip(screen, ct_models, fast, br, bf, 'draw')

                        print("Loop took {:} seconds.".format(time()-start))

                        start = time()

                        cv2.imshow('AimAssistant', process.get_image())

                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break

    def analyse_video(self, path):
        pass