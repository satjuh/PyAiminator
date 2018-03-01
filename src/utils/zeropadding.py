
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
import os
from src.paths import DataPath
import pickle

def zeropadding(image, image_height, image_width):
    width, height = image.size
    
    if height == image_height and width == image_width:
        return image
    
    elif height <= image_height and width <= image_width:
        offset = (int((image_width - width) / 2), int((image_height -height) / 2))
        background = Image.new('RGB', [image_width, image_height],(0,0,0))
        background.paste(image, offset)
        return background
    
    elif height > image_height or width > image_width:
        image.thumbnail((image_height, image_width))
        zeropadding(image, image_height, image_width)
    
if __name__ == '__main__':
    image_height = 720
    image_width = 960
    dirname = 'images'
    results = [];
    images = []
    Dpath = DataPath()
    
    images = []
    for directory in os.listdir(Dpath.collected):
        files = os.path.join(Dpath.collected, directory)
        for file in os.listdir(files):
            with open(os.path.join(files, file), 'rb') as f:
                data = pickle.load(f)
                image = data['image']
                detections = data['detections']
            print("test")
            images.append(image)

    #imshow(images[0])
    
  