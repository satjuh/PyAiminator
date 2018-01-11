import math
from multiprocessing import Pool, cpu_count

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import norm
from regionprobs import RegionProbs as rp


class ImageProcess:

    def __init__(self, image, templates, fast_detector, brisk_detector, bf_detector, *modes):
        """
        Constructor for ImageProcess object.

        :param image:
        :param templates:
        :param fast_detector:
        :param brisk_detector:
        :param bf_detector:
        :modes:
        """
        self.__templates = templates
        self.__fast = fast_detector
        self.__br = brisk_detector
        self.__bf = bf_detector
        self.__image = image
        self.__detected = 0
        self.__modes = modes
        self.__MAX_EUCLIDEAN = math.sqrt(195075)

        # normalizes the screen imgae
        self.__norm = norm.normalize_image(image)
        # grayscale is used in all other analysis
        self.__gray = cv2.cvtColor(self.__norm, cv2.COLOR_RGB2GRAY)
        # blurs the image using median blur method
        self.__blurred = cv2.medianBlur(self.__gray, 7)
        # extracts edges from image using Canny algorithm
        self.__edges = cv2.Canny(self.__blurred, 100, 200)

        self.analyse_image()

    def get_image(self):
        """
        Return the image
        """
        return self.__gray

    def get_detections(self):
        """
        Returns detected objects
        """
        return self.__detections

    def analyse_image(self):
        """
        Analyses the image contours
        """
        stats = rp(self.__edges, mode='outer_full', output='struct').get_properties()
        self.__detections = [x for x in stats if self.process_contour(x) == 1]

        if 'draw' in self.__modes:
            for detection in self.__detections:
                self.draw_box(detection)

    def process_contour(self, contour):
        """
        Process the contour and detect any objects from it.

        :param contour: contour to be processed
        :return: 1 if detected, 0 otherwise
        """

        detection = 0
        # takes the Area of the contour 
        area = contour.area
        #print(area)
        # takes the aspect ratio
        aspect_ratio = contour.aspect_ratio
        #print(area)
        # roughly cuts out the biggest and smallest areas
        if 25000 > area > 50:

            # Save matches with scores to dict so we can name them.
            match = {}
            cnt = contour.cnt
            x, y, w, h = contour.bounding_box
            # draws a new masked image based on contour.
            #print(contour.filled_image.shape)
            #print(self.__gray[x:x+h, y:y+w].shape)
            mask = np.zeros(self.__gray.shape, np.uint8)
            cv2.drawContours(mask, [cnt], 0,255, -1)
            #cv2.fillPoly(mask, pts=[cnt], color=(255,255,255))
            # FAST points of interest analyze on the masked imge.

            masked = cv2.bitwise_and(self.__gray, mask)

            kp = self.__fast.detect(masked[y:y+h,x:x+w], None)
            #plt.imshow(masked[y:y+h,x:x+w], cmap='gray')
            #plt.show()
            GOOD_KP = 35

            if len(kp) > GOOD_KP:

                for template in self.__templates:

                    angle = abs(contour.orientation)
                    # compare the contours values against each templates values
                    ret = cv2.matchShapes(template.cnt, cnt, 1, 0.0)
                    angle_d = abs((angle - template.angle) / template.angle)
                    ar_d = abs((aspect_ratio - template.ar) / template.ar)

                    # sums them up to roughly evaluate the feature matches
                    match_value = ret + ar_d + angle_d

                    name = template.name
                    #print(match_value)
                    # rought estimate to cut out all over the top different objects
                    if 0 < match_value < 6:

                        # takes the average color value in the contours area
                        mean = cv2.mean(self.__norm, mask=mask)

                        kp, des = self.__br.compute(masked[y:y+h,x:x+w], kp)
                        # calculates the matches using the BF matcher.
                        matches = self.__bf.match(template.des, des)

                        # store all the good matches as per Lowe's ratio test.
                        #good = [m for m in matches if m.distance < 0.5]
                        #print(len(matches), len(kp))
                        if len(matches) >= len(kp)*0.25:
                            # calculates the euclidean distance
                            eucli = np.sqrt(
                                (mean[0] - template.mean[0]) ** 2 +(mean[1] - template.mean[1]) ** 2 +(mean[2] - template.mean[2]) ** 2)
                            # compares the calculated value to maximum possible value
                            eucli_d = eucli / self.__MAX_EUCLIDEAN
                            match[name] = eucli_d

                        else:
                            match[name] = 0.6

                    else:
                        match[name] = self.__MAX_EUCLIDEAN

                # sorts the match dict
                sorted_matches = sorted(match, key=lambda x:match[x])

                goods = [match[x] for x in sorted_matches if match[x] < self.__MAX_EUCLIDEAN]

                if len(goods) > 2:
                    print(match[sorted_matches[0]])
                    if 0.1 > match[sorted_matches[0]] >= 0:
                        contour.name = sorted_matches[0]
                        detection += 1

                    # if the best match is within 0 and 0.2 we detect that contour as most similiar to that template and
                    # the second value is also somewhere close.
                    elif 0.20 > match[sorted_matches[0]] >= 0.1:
                        contour.name='m'
                        detection += 1

                    # if it's somewhere between 0.2 and 0.8 it might be but the program can't tell with good enough
                    # certainty so the text is '?'.
                    elif 0.20 <= match[sorted_matches[0]] <= 0.8:
                        contour.name='?'

                    # all the other contours will be ignored.

        return detection

    def draw_box(self, contour):
        """
        Draws the rectangle and text to indicate the detected object.

        :param contour: 
        """
        x, y, w, h = contour.bounding_box

        cv2.rectangle(self.__image, (x, y), (x + w, y + h), (255, 255, 52), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.__image, contour.name, (x - 5, y - 5), font, 0.9, (255, 255, 52), 1)