"""
Author: Da Chen
This script resize all images to be 130 * 130.
"""
import os
import cv2
import glob
import numpy as np
from skimage import io, transform

# Function to resize the input image.
def reSize(img, output_size):
    img = transform.resize(img, (output_size, output_size), mode='constant')
    return img

# Get all image names.
img_ok = glob.glob("./newimages/*ok*.png")
img_stop = glob.glob("./newimages/*stop*.png")
img_punch = glob.glob("./newimages/*punch*.png")
img_peace = glob.glob("./newimages/*peace*.png")
img_nothing = glob.glob("./newimages/*nothing*.png")

for img_path in img_ok:
    img = io.imread(img_path)
    resized_img = cv2.resize(img, (130, 130))
    cv2.imwrite(img_path, cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
