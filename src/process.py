import math
import os
import pickle
import shutil
import sys
from time import sleep, time

import cv2
import numpy as np
import pandas as pd
from PIL import ImageGrab

from src.utils.norm import normalize_image
from src.paths import DataPath
from src.regionprobs import RegionProbs as rp
from src.templates import make_templates


class ImageProcess:

    def __init__(
        self, 
        image, 
        templates, 
        fast_detector, 
        brisk_detector, 
        bf_detector,
        *args,
        **kwargs):
        """
        Constructor for ImageProcess object.

        :param image: image to be analysed
        :param templates: templates that we try to find from the image
        :param fast_detector: cv2 Fast feature detector
        :param brisk_detector: cv2 BRISK keypoint feature analyser
        :param bf_detector: cv2 BruteForce feature matcher
        """
        super().__setattr__('__dict__', {})
        self.image = image

        self.__templates = templates
        self.__fast = fast_detector
        self.__br = brisk_detector
        self.__bf = bf_detector
        self.__detected = 0
        self.__modes = args
        self.__kwargs = kwargs
        self.__MAX_EUCLIDEAN = math.sqrt(195075)

        self.process_image()
        self.analyse_image()

    def process_image(self):

        norm = normalize_image(self.image)
        self.__norm = norm

        gray = cv2.cvtColor(self.__norm, cv2.COLOR_RGB2GRAY)
        self.__gray = gray

        blurred1 = cv2.medianBlur(self.__gray, 11)
        self.__blurred = blurred1

        kernel = np.ones((130,130), np.uint8)
        opening1 = cv2.morphologyEx(self.__blurred, cv2.MORPH_BLACKHAT, kernel)

        threshold1 = cv2.inRange(opening1, 100, 170)
        masked1 = cv2.bitwise_and(self.__blurred, threshold1)

        opening2 = cv2.morphologyEx(masked1, cv2.MORPH_TOPHAT, kernel)
        blurred2 = cv2.GaussianBlur(opening2, (15,15), 3)

        threshold2 = cv2.inRange(blurred2, 5,130)
        #masked2 = cv2.bitwise_and(self.__blurred, threshold2)

        self.__bw = threshold2

        if 'debug' in self.__kwargs:
            self.test = eval(self.__kwargs['debug'])

    def __getattr__(self, key):
        """
        Gets class attribute
        Raises AttributeError if key is invalid
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError

    def __setattr__(self, key, value):
        """
        Sets class attribute according to value
        If key was not found, new attribute is added
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            super().__setattr__(key, value)

    def analyse_image(self):
        """
        Analyses the image contours
        """
        contours = rp(self.__bw).get_properties()
        self.detections = [x for x in contours if self.process_contour(x) == 1]

        if 'collect' in self.__modes:
            self.detection_data = np.array([x.image(self.image) for x in self.detections])

        if 'draw' in self.__modes:
            for detection in self.detections:
                self.draw_box(detection)

    def process_contour(self, contour):
        """
        Process the contour and detect any objects from it.

        :param contour: contour to be processed
        :return: 1 if detected, 0 otherwise
        """
        try:
            detection = 0

            area = contour.area
            # roughly cuts out the biggest and smallest areas
            if 10000 > area > 500:

                # Save matches with scores to dict so we can name them.
                match = {}
                cnt = contour.cnt
                x, y, w, h = contour.bounding_box

                # draws a new masked image based on contour.
                mask = np.zeros(self.__gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], 0,255, -1)

                masked = cv2.bitwise_and(self.__gray, mask)

                # FAST points of interest analyze on the masked imge.
                kp = self.__fast.detect(masked[y:y+h, x:x+w], None)

                if len(kp) > 35:

                    for template in self.__templates:
                        aspect_ratio = contour.aspect_ratio
                        angle = abs(contour.orientation)
                        # compare the contours values against each templates values
                        ret = cv2.matchShapes(template.cnt, cnt, 1, 0.0)
                        angle_d = abs((angle - template.angle) / template.angle)
                        ar_d = abs((aspect_ratio - template.ar) / template.ar)

                        # sums them up to roughly evaluate the feature matches
                        match_value = ret + ar_d + angle_d

                        name = template.name
                        # rought estimate to cut out all over the top different objects
                        if 0 < match_value < 2:

                            # takes the average color value in the contours area
                            mean = cv2.mean(self.__norm, mask=mask)

                            # extract the features using brisk
                            kp, des = self.__br.compute(masked[y:y+h, x:x+w], kp)
                            # calculates the matches using the BF matcher.
                            matches = self.__bf.match(template.des, des)

                            # store all the good matches as per Lowe's ratio test.
                            if len(matches) >= len(kp)*0.65:
                                # calculates the euclidean distance
                                eucli = np.sqrt(
                                    (mean[0] - template.mean[0]) ** 2 +(mean[1] - template.mean[1]) ** 2 +(mean[2] - template.mean[2]) ** 2)
                                # compares the calculated value to maximum possible value
                                eucli_d = eucli / self.__MAX_EUCLIDEAN
                                match[name] = eucli_d
                            else:
                                match[name] = 0.6
                        else:
                            match[name] = 1

                    # sorts the match dict
                    sorted_matches = sorted(match, key=lambda x:match[x])

                    goods = [match[x] for x in sorted_matches if match[x] < 1]

                    if len(goods) > 2:
                        # Checks the best match percentage and choose the name accordingly
                        if 0.1 > match[sorted_matches[0]] >= 0:
                            contour.name = sorted_matches[0]
                            detection += 1
                        elif 0.20 > match[sorted_matches[0]] >= 0.1:
                            contour.name='m'
                            detection += 1
                        elif 0.20 <= match[sorted_matches[0]] <= 0.8:
                            detection += 1
                        # all the other contours will be ignored.
            return detection

        except cv2.error as e:
            print(e)
            return detection

    def draw_box(self, contour):
        """
        Draws the rectangle and text to indicate the detected object.

        :param contour: 
        """
        x, y, w, h = contour.bounding_box

        cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 255, 52), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, contour.name, (x - 5, y - 5), font, 0.9, (255, 255, 52), 1)


class CollectProcess:
    """
    Collect Process class to harvest data
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor for CollectProcess class.
        Initialize the fast, brisk and bf detectors. Loads the templates data and
        create the directory for collection.

        :param args:
        :param kwargs:
        """
        self.__fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=50)
        self.__br = cv2.BRISK_create()
        self.__bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        try:
            model_type = kwargs['model_type']
        except KeyError:
            model_type = 'CT'

        self.__templates = make_templates(model_type, self.__fast, self.__br)

        self.__modes = args
        self.__kwargs = kwargs

        self.__dp = DataPath()

        self.__file_index = self.__dp.get_index('collected')
        self.__dir = os.path.join(self.__dp.collected, str(self.__file_index))
        os.mkdir(self.__dir)

        self.__index = 0

        col_names = ('detections', 'intensity', 'fast_kp', 'process_time')
        self.__df = pd.DataFrame(columns=col_names)

    def info(self):
        if self.__df.empty:
            return "\nError! No data was collected"
        else:
            det_count = sum(self.__df['detections'].tolist())
            mean_inten = np.mean(self.__df['intensity'].tolist())
            mean_time = np.mean(self.__df['process_time'].tolist())
            return("\nProcess completed\n\n"
                   "Collected total of {:d} samples\n"
                   "Detection count: {:d}\n"
                   "Average intensity: {:.5f}\n"
                   "Average time: {:.5f}\n".format(
                       self.__index,
                       det_count,
                       mean_inten,
                       mean_time))

    def collect_from_screen(self, width, heigth, windowed=True, time_out=1):

        self.__time_out = time_out

        if width <= 0 or heigth <= 0:
            raise ValueError("Incorrect dimensions for screen capture")

        # if on windowed mode, lower the capture area
        if windowed:
            padding = 27
            heigth += padding

        if 'debug' in self.__modes:
            while True:
                # captures the screen with ImageGrab in RGB.
                screen = np.array(ImageGrab.grab(bbox=(0,padding,width,heigth)))
                # converts it to BGR
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                process = self.analyse_frame(screen)

                cv2.imshow('AimAssistant', process.image)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        else:
            while True:
                screen = np.array(ImageGrab.grab(bbox=(0,padding,width,heigth)))
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                process = self.analyse_frame(screen)
                if 'lim' in self.__kwargs:
                    if self.__index >= self.__kwargs['lim']:
                        break

        print(self.info())
        if not self.__df.empty:
            csv_name = 'process_{:d}.csv'.format(self.__file_index)
            self.__df.to_csv(os.path.join(self.__dp.dataframe, csv_name))
        else:
            shutil.rmtree(self.__dir)

    def collect_from_video(self, file, frame_limit):
        """
        """
        if not os.path.exists(os.path.join(self.__dp.video, file)):
            raise OSError

        # Open the video file
        cap = cv2.VideoCapture(path)
        if 'debug' in self.__modes:
            while cap.isOpened():
                # Read the frame
                ret, frame = cap.read()
                process = self.analyse_frame(frame)

                cv2.imshow('AimAssistant', process.image)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        else:
            while cap.isOpened():
                # Read the frame
                ret, frame = cap.read()
                process = self.analyse_frame(frame)

        #TODO saving the data part.

    def analyse_frame(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        intensity = cv2.mean(gray)[0]
        kp = self.__fast.detect(gray, None)

        process_time = time()
        process = ImageProcess(image, self.__templates, self.__fast, self.__br, self.__bf, 'collect')
        process_time = time() - process_time

        if process.detections:
            print("Detected.")
            new_row = [len(process.detections), intensity, len(kp), process_time]

            self.__df.loc[self.__index] = new_row
            # Save the data into packet for further use
            packet = {'image':image, 'detections':process.detection_data}

            file_name = os.path.join(self.__dir, "{:d}.pickle".format(self.__index))
            with open(file_name, 'wb') as file:
                pickle.dump(packet, file)

            self.__index += 1

            # Sleep so we don't analyse the same frame multiple times
            sleep(self.__time_out)

        return process
