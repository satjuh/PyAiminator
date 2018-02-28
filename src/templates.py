import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import src.utils.norm as norm
from src.paths import DataPath
from src.regionprobs import RegionProbs as rp


class Template:

    def __init__(self, img, name, fast_detector, brisk_detector):
        """
        Template class constructor
        :param img:
        :param name:
        :param fast_detector:
        :param brisk_detector:
        """
        super().__setattr__('__dict__', {})
        self.name = name
        self.img = img

        self.__fast = fast_detector
        self.__br = brisk_detector

        self.analyse()

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

    def analyse(self):
        
        norm_img = norm.normalize_image(self.img)
        gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)

        mask = cv2.inRange(gray, 20, 100)
        masked = cv2.bitwise_and(gray, mask)

        edges = cv2.Canny(blurred, 100, 200)

        # saves the templates descriptors
        kp = self.__fast.detect(masked, None)
        kp, des = self.__br.compute(masked, kp)

        contours = rp(bw=edges).get_properties()

        # Pick the biggest contour aka the outlines of the template
        contours = sorted(contours, key=lambda x: x.area, reverse=True)
        contour = contours[0]

        self.des = des
        self.cnt = contour.cnt
        self.ar = contour.aspect_ratio
        self.mean = cv2.mean(norm_img, mask=mask)
        self.intensity = cv2.mean(gray, mask=mask)
        self.angle = contour.orientation


def make_templates(model_type, fast, br):
    
    p = DataPath()
    path = os.path.join(p.templates, model_type)
    templates = []
    for x in os.listdir(path):
        name = x.split(".")[0]
        original = cv2.imread(os.path.join(path, x), 1)
        templates.append(Template(original, name, fast, br))

    return templates
