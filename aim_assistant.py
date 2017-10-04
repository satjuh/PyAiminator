import math
import os
import sys
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageGrab

# initialises fastfeaturedetector to get initial points of interests
fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=50)

# initialises Binary Robust Invariant Scalable Keypoints for 
# keypoint descriptor analyze
br = cv2.BRISK_create()

# BruteForce matcher to compare matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def normalize_image(img):
    """
    Normalizes the BGR image

    :param img: image that we want to normalize (needs to be bgr format)
    :return: normalized bgr image
    """
    # creates two identical empty pictures
    norm = np.zeros(img.shape, dtype=np.float32)
    norm_bgr = np.zeros(img.shape, dtype=np.uint8)

    # reads the b g r values from rgb picture
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    # counts the total value
    total = r + g + b

    # normalizes every value and sets values based on example: r' = r/total * 255.0
    np.seterr(divide='ignore', invalid='ignore')
    norm[:,:,0] = b/total * 255.0
    norm[:,:,1] = g/total * 255.0
    norm[:,:,2] = r/total * 255.0
    
    norm_bgr = cv2.convertScaleAbs(norm)

    return norm_bgr


def detection(contours, img, text):
    """
    Draws the rectangle and text to the detected object.

    :param contours: can be only one or many.
    :param img: image we draw into
    :param text: text we want to draw over the rectangle
    """
    for cnt in contours:
        
        x,y,w,h = cv2.boundingRect(cnt)
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,52),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (x-5,y-5), font, 0.9, (255,255,52), 1)
        #print('{} detected at ({:.1f},{:.1f})'.format(text,x,y))

def make_templates(path):
    
    templates = {}
    
    for x in os.listdir(path):
        
        # reads the original RGB template image
        original = cv2.imread(path + x, 1)
        # normalises the image
        model_img = normalize_image(original)
        # makes a grayscale image
        model_gray = cv2.cvtColor(model_img,cv2.COLOR_BGR2GRAY)
        # makes a rough mask of the image
        model_mask = cv2.inRange(model_gray, 0, 100)
        # fills the mask with gray image
        masked = cv2.bitwise_and(model_gray, model_mask)
        # takes the outer edges of the mask with gradient morph
        kernel = np.ones((3,3),np.uint8)
        gradient = cv2.morphologyEx(model_mask, cv2.MORPH_GRADIENT, kernel)
        # makes a data set of template_name : [images] for further usage
        templates[x.split(".")[0]] = [masked, model_mask, model_gray, gradient, model_img, original]

    return templates


def analyze_templates(templates):

    data = {}
    for template in templates:
        sub_data = {}

        # saves the templates descriptors
        kp = fast.detect(templates[template][0], None)
        kp, des = br.compute(templates[template][0], kp)
        sub_data['des'] = des

        bw2, contours, hierarchy = cv2.findContours(templates[template][3], 
                                                    cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_NONE)
        # sometimes we might encounter small areas that are not part of the mask so we delete them.
        for count, cnt in enumerate(contours):
            if len(cnt) < 20:
                del contours[count]

        # saves the contour data
        cnt = contours[0]
        sub_data['cnt'] = cnt

        x,y,w,h = cv2.boundingRect(cnt)
        
        # aspect ratio for further use        
        aspect_ratio = float(w)/h
        sub_data['ar'] = aspect_ratio

        # checks the average color of the object
        mean_val = cv2.mean(templates[template][4],mask = templates[template][1])
        sub_data['mv'] = mean_val
        
        # Centroid, (Major Axis, minor axis), orientation
        # saves the angle for further use
        (x, y), (maxa, mina), angle = cv2.fitEllipse(cnt)
        sub_data['angle'] = angle
        
        # adds all data of the contour to the dataset
        data[template] = sub_data
    
    return data


def contour_analyse(bw,img,gray,norm_img):
    """
    Measures of properties of images contours and highlights them into the
    displayed image.

    :param bw: binary threshold image
    :param img: image we want to draw rectangles into.
    """
    # finding the contours and only returns the outer contours.
    # we only use contours and ignore hierarchy and bw2.
    bw2, contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # handle every contour in the image.
    for cnt in contours:
        
        # cv2.fitEllipse needs at least 5 points to work.
        if len(cnt) > 5:
            
            # takes the Area of the contour 
            m = cv2.moments(cnt)
            area = m['m00']
            
            # Centroid, (Major Axis, minor axis), orientation
            (x, y), (maxa, mina), angle = cv2.fitEllipse(cnt)
            
            # finds the aspect ratio with boundingrect
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            # roughly cuts out the biggest and smallest areas
            if 10000 > area > 500:
                
                match = {}
                
                # draws a new masked image based on contour.
                mask = np.zeros(gray.shape,np.uint8)
                cv2.drawContours(mask,[cnt],0,255,-1)
                
                masked = cv2.bitwise_and(gray,mask)
                kp = fast.detect(masked,None)

                # maximum euclidean value that we use to evaluate contours.
                # calculated with : sqrt(delta(R)^2 + delta(G)^2 + delta(B)^2)
                # and the maximum value is with black(0,0,0) and white(255,255,255)
                max_euclidean= math.sqrt(195075)

                if len(kp) > 35:
                    for template in templates_data:
                        # fetch the wanted data from template data
                        angle_ = templates_data[template]['angle']
                        cnt_ = templates_data[template]['cnt']
                        aspect_ratio_ = templates_data[template]['ar']

                        # compare the contours values against each templates values
                        ret = cv2.matchShapes(cnt_,cnt,1,0.0)
                        angle_d = abs((angle - angle_) / angle_)
                        ar_d = abs((aspect_ratio - aspect_ratio_) / aspect_ratio_)
                        
                        # sums them up to roughly evaluate the feature matches
                        match_value = ret + ar_d + angle_d
                        
                        # rought estimate to cut out all over the top different objects
                        if 0 < match_value < 3:
                            
                            # takes the average color value in the contours area
                            mean_val = cv2.mean(norm_img, mask=mask)
                            mean_val_ = templates_data[template]['mv']

                            des1 = templates_data[template]['des']
                            kp, des2 = br.compute(img, kp)
                            # calculates the matches using the BF matcher.
                            matches = bf.match(des2,des2)
                            
                            # store all the good matches as per Lowe's ratio test.
                            good = [m for m in matches if m.distance < 0.7]

                            if len(good) > 35:
                                # calculates the euclidean distance
                                eucli = math.sqrt(
                                    (mean_val[0] - mean_val_[0]) ** 2 +(mean_val[1] - mean_val_[1]) ** 2 +(mean_val[2] - mean_val_[2]) ** 2)
                                
                                # compares the calculated value to maximum possible value
                                eucli_d = eucli / max_euclidean
                                match[template] = eucli_d

                            else:
                                match[template] = 0.6

                        else:
                            match[template] = max_euclidean
                    
                    # sorts the match dict
                    sorted_matches = sorted(match,key=lambda x:match[x],reverse=False)

                    goods = [match[x] for x in sorted_matches if match[x] < max_euclidean]
                    if len(goods) > 2:
                        if 0.1 > match[sorted_matches[0]] >= 0:
                            text = sorted_matches[0]
                            detection([cnt], img,text)

                        # if the best match is within 0 and 0.2 we detect that contour as most similiar to that template and
                        # the second value is also somewhere close.
                        elif 0.20 > match[sorted_matches[0]] >= 0.1:
                            text = 'm'
                            detection([cnt], img,text)

                        # if it's somewhere between 0.2 and 0.8 it might be but the program can't tell with good enough
                        # certainty so the text is '?'.
                        elif 0.20 <= match[sorted_matches[0]] <= 0.8:
                            text = '?'
                            detection([cnt],img,text)

                        # all the other contours will be ignored.


def roi(img, vertices):
    """
    Masks the image according to vertices and creates the sort of "Region of interest"

    :param img: image we fill the region with
    :param vertices: points that determine the region of interest
    :return: masked image 
    """
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,np.int32([vertices]),255)
    
    return cv2.bitwise_and(img,mask)


def process_image(original,type):
    """
    Process the RGB image to filter out the background and detect the models

    :param original: RGB image
    :return: processed BGR image
    """
    start = time()
    # converts the image into BGR format for cv2 to display later
    # this is the img we also draw the rectangles into
    img = cv2.cvtColor(original,cv2.COLOR_RGB2BGR)
    
    # normalizes the screen imgae
    norm_img = normalize_image(img)

    # grayscale is used in all other analysis
    grayscale = cv2.cvtColor(norm_img, cv2.COLOR_RGB2GRAY)

    # makes a rough estimation if the image is dark or light
    img_mean = cv2.mean(grayscale)
    # the case if the image is lighter
    if img_mean[0] > 120:
        
        # blurs the image using median blur method
        blur_ = cv2.medianBlur(grayscale,7)

        # morphological function to filterout the background and bring
        # front the darker areas - thesis is: all the models are darker than
        # their backgrounds
        kernel_ = np.ones((150,150),np.uint8)
        opening_ = cv2.morphologyEx(blur_,cv2.MORPH_BLACKHAT,kernel_)

        # makes the first rough mask of the image and applies it to the blurred image
        mask = cv2.inRange(opening_,100,170)
        masked_ = cv2.bitwise_and(blur_,mask)

        # applies the tophat method into masked image
        opening = cv2.morphologyEx(masked_,cv2.MORPH_TOPHAT,kernel_)

        # makes a gaussian blur to smooth the opening
        blur = cv2.GaussianBlur(opening,(21,21),3)

        # makes a threshold to detect the models
        threshold = cv2.inRange(blur,5,130)
        erode = cv2.erode(threshold,(15,15),iterations=3)

        # takes the region of interest
        vertices_ = np.array([[0,0],[800,0],[800,334],[480,334],[480,470],[200,527],[0,527]])
        
        if type == 'img':
            # finds the contours and draws the rectangles
            contour_analyse(erode, img, grayscale, norm_img)
        return eval(type)
    
    # the case if the image is darker
    else:  
        grayscale_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # blurs the image using median blur method
        blur_ = cv2.medianBlur(grayscale_,7)

        # morphological function to filterout the background and bring
        # front the darker areas - thesis is: all the models are darker than
        # their backgrounds
        kernel_ = np.ones((200,200),np.uint8)
        opening_ = cv2.morphologyEx(blur_,cv2.MORPH_BLACKHAT,kernel_)

        # makes the first rough mask of the image and applies it to the blurred image
        mask = cv2.inRange(opening_,40,120)
        masked_ = cv2.bitwise_and(blur_, mask)

        erode = cv2.inRange(masked_,8,33)

        # takes the region of interest
        vertices_ = np.array([[0,0],[800,0],[800,334],[480,334],[480,470],[200,527],[0,527]])
        
        if type == 'img':
            # finds the contours and draws the rectangles
            contour_analyse(erode, img, grayscale, norm_img)
        return eval(type)


def examples():
    
    example_path = './images/examples/'
    for file in os.listdir(example_path):

        img = cv2.cvtColor(cv2.imread(example_path + file,1),cv2.COLOR_BGR2RGB)
        plt.imshow(cv2.cvtColor((process_image(img,'img')),cv2.COLOR_BGR2RGB))
        plt.title(file.split('.')[0])
        plt.show()


def main():
    """
    Main function of the aimbot.py - captures the screen of the size 800x600 in
    the top-left corner of the screen and feeds a new window to display with detected
    models
    """
    # disgusting global variables.
    global ct_models
    global templates_data
    
    # initialize the data of the templates
    ct_models = make_templates('images/templates/CT/')
    templates_data = analyze_templates(ct_models)
    
    print('Hello there!\n')
    while True:
        available = {'norm':'norm_img','gray':'grayscale',
                    'blur':'blur_','morph':'opening_',
                    '1 mask':'mask','masked':'masked_',
                    '2 mask':'erode','full':'img'}
        
        mode = input('Enter mode: ')

        try:
            start = time()
            t = []
            
            if mode == "":
                print('Goodbye!')
                break
            elif mode == '!examples':
                examples()
                continue
            else:
                output = available[mode]
            
            while True:
                
                # captures the screen.
                screen = np.array(ImageGrab.grab(bbox=(0,27,800,627)))

                # process the screen.
                new_screen = process_image(screen,output)

                # calculates the frames per second of the loop.
                t.append(time() - start)
                if sum(t) > 1:
                    print('FPS: {:.1f}'.format(len(t)/sum(t)))
                    t.clear()
                start = time()

                cv2.imshow('AimAssistant', new_screen)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        except KeyError:
            print("Invalid mode.")


if __name__ == '__main__':
    main()
