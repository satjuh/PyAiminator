import math
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import checks 


class ImgError(Exception):
    """
    """
    def __init__(self,message,error):
        
        print(message+'\n',error)


class RegionProbs:
    """
    """
    def __init__(self, bw, *properties, mode='outer_full',output='struct'):
        """

        :param bw:
        :param properties:
        :param mode:
        :param output:
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
                raise ImgError('Wrong type of image', 'not bw')
            else:
                self.__bw = bw

            if not isinstance(mode,str) or mode not in self.__modes:
                raise KeyError
            else:
                self.__mode = mode

            initial = [c for c in properties if c in property_list]
            if len(initial) == 0:
                initial = ['area', 'centroid', 'bounding_box']
            elif 'all' in initial:
                initial = property_list
                initial.pop(initial.index('all'))
            self.__properties = initial
            
            if output in self.__outputs:
                self.__output = output
            else:
                self.__output = self.__outputs[0]

            zeros = np.where( bw == 0)
            ones = np.where(bw == 1)

            self.__img, self.__contours = self.extract()

        except KeyError:
            print('Invalid mode.')
            s = ', '
            print('Available modes:\n',s.join(self.__modes.keys()))
        
        except ImgError:
            print()

    def extract(self):
        """

        :return:
        """
        retr = eval(self.__modes[self.__mode]['retr'])
        approx = eval(self.__modes[self.__mode]['approx'])

        # extract contours and hierarchy.
        bw2, contours, hierarchy = cv2.findContours(self.__bw, retr, approx)

        if self.__mode == "outer_simple" or self.__mode == 'outer_full':
            data = []
            for count, cnt in enumerate(contours):
                data.append(Contour(cnt, count))
            
            return bw2, np.array(data)
        
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
                    data.append(Contour(cnt, count))
                elif parent < 0:
                    # I'm the first in my family!
                    data.append(Contour(cnt, count, child=child))
                elif child < 0:
                    # I'm the youngest child.
                    data.append(Contour(cnt, count, parent=parent))
                else:
                    # I'm in the middle!
                    data.append(Contour(cnt, count, child=child, parent=parent))

            return bw2, np.array(data)

    def get_properties(self):
        """

        :return:
        """
        try:    
            if self.__output == self.__outputs[0]:
                return self.__contours

            elif self.__output == self.__outputs[1]:

                data = {}
                for prop in self.__properties:
                    data[prop] = [x.__get__(prop) for x in self.__contours]

                return pd.DataFrame(data)
        
        except AttributeError:
            return []


# TODO: convert all the math. to np.
class Contour(RegionProbs):
    """

    """
    def __init__(self, cnt, my_number, child=False, parent=False):
        """

        :param cnt:
        :param my_number:
        :param child:
        :param parent:
        :return:
        """
        self.__cnt = cnt

        self.__moment = cv2.moments(cnt)

        self.__number = my_number
        self.__child = child
        self.__parent = parent

    def __get__(self, var):
        """

        :param var:
        :return:
        """
        return getattr(self, var)

    def pointer(self, param):
        """

        :param param:
        :return:
        """
        if param == 'child':
            return self.__child
        elif param == 'parent':
            return self.__parent
        else:
            return False

    @property
    def cnt(self):
        return self.__cnt

    def get_number(self):
        """

        :return:
        """
        return self.__number
    
    @property
    def area(self):
        """

        :return:
        """
        return self.__moment['m00']

    @property
    def aspect_ratio(self):
        """

        :return:
        """        
        x, y, w, h = self.bounding_box

        return float(w)/h
    
    @property
    def bounding_box(self):
        """

        :return:
        """
        return cv2.boundingRect(self.__cnt)
    
    @property
    def centroid(self):
        """

        :return:
        """
        cx = self.__moment['m10'] / self.__moment['m00']
        cy = self.__moment['m01'] / self.__moment['m00']

        return (cx, cy)

    @property
    def convex_area(self):
        """

        :return:
        """
        hull = self.convex_hull
        
        return cv2.contourArea(hull)
    
    @property
    def convex_hull(self):
        """

        :return:
        """
        return cv2.convexHull(self.__cnt)
    
    def convex_image(self, image):
        """

        :param image:
        :return:
        """
        pass
    
    @property
    def eccentricity(self):
        """

        :return:
        """
        pass

    @property
    def equiv_diameter(self):
        """

        :return:
        """
        return np.sqrt(4 * self.area / math.pi)
    
    @property
    def extent(self):
        """

        :return:
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

    def filled_image(self, image):
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
        
        if len(self.__cnt) > 5:
            (x,y), (major_axis, minor_axis), angle = cv2.fitEllipse(self.__cnt)
        else:
            major_axis = 0

        return major_axis

    @property
    def minor_axis_len(self):
        
        if len(self.__cnt):
            (x,y), (major_axis, minor_axis), angle = cv2.fitEllipse(self.__cnt)
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
            
            du20 = self.__moment['m20'] / self.__moment['m00'] - x**2 
            du11 = self.__moment['m11'] / self.__moment['m00'] - x*y 
            du02 = self.__moment['m02'] / self.__moment['m00'] - y**2 
            
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



def main():
    img = cv2.imread('C:/Users/Eemeli/Documents/Projects/PyAiminator/images/examples/circlesBrightDark.png', 0)
    bw = cv2.inRange(img,0,50)
    plt.imshow(bw,cmap='gray'),plt.show()
   # moment(bw)
    stats = RegionProbs(bw,'all', mode='hier_full',output='table')
    
    data = stats.get_properties()
    print(data)
    #for i in data:
        #print(i.pointer('child'))
        #print(i.major_axis_len)
        #print(i.equiv_diameter)
        #print(i.area)
        #print(i.eccentricity)
        #print(i.centroid)
        #print(i.orientation)
        #print(i.extrema)
    #print(time()-start)

if __name__ == '__main__':
    main()