
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
import os
from src.paths import DataPath
import pickle

"""
    Pads the image with zeros that it can be fed to the network
"""
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

  