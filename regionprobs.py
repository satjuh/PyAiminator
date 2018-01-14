import math
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import checks


class RegionProbs:
    """
    Contour analyze class emulating the way Matlabs Regionprobs works.
    """
    def __init__(self, bw, *properties, mode='outer_full',output='struct'):
        """
        Constructor for Regionprobs

        :param bw: bitwise image
        :param properties: properties to be used
        :param mode: mode for contour detection
        :param output: output type.
        """
        self.__modes = {'outer_simple':{'retr':'cv2.RETR_EXTERNAL', 'approx':'cv2.CHAIN_APPROX_SIMPLE'},
                        'outer_full':{'retr':'cv2.RETR_EXTERNAL', 'approx':'cv2.CHAIN_APPROX_NONE'},
                        'hier_simple':{'retr':'cv2.RETR_CCOMP', 'approx':'cv2.CHAIN_APPROX_SIMPLE'},
                        'hier_complex':{'retr':'cv2.RETR_CCOMP', 'approx':'cv2.CHAIN_APPROX_NONE'},
                        'hier_full':{'retr':'cv2.RETR_TREE', 'approx':'cv2.CHAIN_APPROX_NONE'}}

        property_list = ['area', 'aspect_ratio', 'bounding_box', 'centroid', 'convex_area', 'eccentricity',
                         'equiv_diameter', 'extent', 'extrema', 'major_axis_len', 'minor_axis_len', 'orientation',
                         'perimeter', 'solidity', 'all']

        self.__outputs = ['struct', 'table']

        try: 
            if not checks.check_bit_array(bw):
                raise ValueError
            else:
                self.__bw = bw

            if not isinstance(mode, str) or mode not in self.__modes:
                raise KeyError
            else:
                self.__mode = mode

            # looks up the valid properties
            initial = [c for c in properties if c in property_list]
            # if there's none - returns the default
            if len(initial) == 0:
                initial = ['area', 'centroid', 'bounding_box']

            elif 'all' in initial:
                initial = property_list
                initial.pop(initial.index('all'))

            self.__properties = initial

            # checks if the output mode is valid
            if output in self.__outputs:
                self.__output = output

            # else uses the default
            else:
                self.__output = self.__outputs[0]

            zeros = np.where( bw == 0)
            ones = np.where(bw == 1)

            self.__img, self.__contours = self.extract()

        except KeyError:
            print('Invalid mode.')
            print('Available modes:\n',  ', '.join(self.__modes.keys()))

        except ValueError:
            print("Wrong image format.")

    def extract(self):
        """
        Extracts the contours from the bw.

        :return: np array of contour objects.
        """
        #print(self.__mode)
        retr = eval(self.__modes[self.__mode]['retr'])
        approx = eval(self.__modes[self.__mode]['approx'])

        # extract contours and hierarchy.
        bw2, contours, hierarchy = cv2.findContours(self.__bw, retr, approx)

        if self.__mode == "outer_simple" or self.__mode == 'outer_full':
            data = []
            for count, cnt in enumerate(contours):
                c = Contour(cnt, self.__bw, count)
                data.append(c)
            return bw2, data

        else:
            data = []
            for count, component in enumerate(zip(contours, hierarchy[0])): 
                cnt = component[0]

                # hierarchy structure: [Next, Previous, First_Child, Parent]
                hier = component[1]
                parent = hier[3]
                child = hier[2]

                if parent < 0 and child < 0:
                    # I'm a lonely wolf!
                    data.append(Contour(cnt, self.__bw, count))
                elif parent < 0:
                    # I'm the first in my family!
                    data.append(Contour(cnt, self.__bw, count, child=child))
                elif child < 0:
                    # I'm the youngest child.
                    data.append(Contour(cnt, self.__bw, count, parent=parent))
                else:
                    # I'm in the middle!
                    data.append(Contour(cnt, self.__bw, count, child=child, parent=parent))

            return bw2, np.array(data)

    def get_properties(self):
        """
        Gets the properties in the output format.

        :return: properties of the contours in wanted output format.
        """
        if self.__output == self.__outputs[0]:
            return self.__contours

        elif self.__output == self.__outputs[1]:
            # do a pandas table of the contours.
            data = {}
            for prop in self.__properties:
                data[prop] = [x[prop] for x in self.__contours]

            return pd.DataFrame(data)


class Contour(RegionProbs):
    """
    Subclass for Regionprobs, a single contour item.
    """
    def __init__(self, cnt, my_number, child=False, parent=False):
        """
        :param cnt: Np array of outline points for contour 
        :param my_number: number that contour can be pointed to. 
        :param child: one of the contours possbile childrens number
        :param parent: contours parent number
        """
        super().__setattr__('__dict__', {})
        self.__dict__['number'] = my_number
        self.__dict__['cnt'] = cnt
        self.__dict__['moment'] = cv2.moments(cnt)
        self.__dict__['child'] = child
        self.__dict__['parent'] = parent
        self.__dict__['name'] = "?"

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
    
    def __add__(self, contour):
        """
        Appends the existing contour array
        """
        self.cnt = np.append(self.cnt, contour.cnt)

    @property
    def area(self):
        """
        Contours area calculated by: #TODO

        :return: area (float)
        """
        return self.moment['m00']

    @property
    def aspect_ratio(self):
        """
        Aspect ratio that covers the contour.

        :return: aspect ratio (float)
        """        
        x, y, w, h = self.bounding_box

        return float(w)/h

    @property
    def bounding_box(self):
        """
        Bounding box that covers contour.

        :return: x,y-cordinates and width,heigth 
        """
        return cv2.boundingRect(self.cnt)

    @property
    def centroid(self):
        """
        Centroid moment of the contour.
        Calculated with: #TODO

        :return: (centroid_x, centroid_y)
        """
        cx = self.moment['m10'] / self.moment['m00']
        cy = self.moment['m01'] / self.moment['m00']

        return (cx, cy)

    @property
    def convex_area(self):
        """
        Area of the hull of the contour.
        Calculated with: #TODO

        :return: hull area (float)
        """
        hull = self.convex_hull

        return cv2.contourArea(hull)

    @property
    def convex_hull(self):
        """
        #TODO
        """
        return cv2.convexHull(self.cnt)

    def convex_image(self, image):
        #TODO
        pass
    
    @property
    def eccentricity(self):
        #TODO
        pass

    @property
    def equiv_diameter(self):
        """
        #TODO
        """
        return np.sqrt(4 * self.area / math.pi)
    
    @property
    def extent(self):
        """
        #TODO
        """
        x, y, w, h = self.bounding_box
        return self.area / (w * h)
    
    @property
    def extrema(self):
        """

        :return:
        """
        cnt = self.__cnt
        
        left = tuple(cnt[cnt[:,:,0].argmin()][0])
        right = tuple(cnt[cnt[:,:,0].argmax()][0])
        top = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottom = tuple(cnt[cnt[:,:,1].argmax()][0])

        return np.array([top,right,bottom,left])

    @property
    def filled_area(self):
        """

        :return:
        """
        pass

    @property
    def filled_image(self):
        """

        :return:
        """
        pass

    def image(self, image):
        """

        :return:
        """
        pass

    @property
    def major_axis_len(self):
        """

        :return:
        """
        if len(self.__cnt) > 5:
            (x,y), (major_axis, minor_axis), angle = cv2.fitEllipse(self.cnt)
        else:
            major_axis = 0

        return major_axis

    @property
    def minor_axis_len(self):
        """

        :return: 
        """
        if len(self.__cnt):
            (x,y), (major_axis, minor_axis), angle = cv2.fitEllipse(self.cnt)
        else:
            minor_axis = 0

        return minor_axis

    @property
    def orientation(self):
        """

        :return:
        """
        x, y = self.centroid
        try:
            
            du20 = self.moment['m20'] / self.moment['m00'] - x**2 
            du11 = self.moment['m11'] / self.moment['m00'] - x*y 
            du02 = self.moment['m02'] / self.moment['m00'] - y**2 
            
            angle = 0.5 * np.arctan2((2 * du11), (du20 - du02))
    
        except ZeroDivisionError:
            angle = 0.0

        return np.degrees(angle)

    @property
    def perimeter(self):
        """

        :return:
        """
        return cv2.arcLength(self.__cnt, closed=True)

    @property
    def pixel_idx_list(self):
        """

        :return:
        """
        pass

    @property
    def pixel_list(self):
        """

        :return:
        """
        return self.__cnt

    @property
    def solidity(self):
        """

        :return:
        """
        return float(self.area) / self.convex_area
