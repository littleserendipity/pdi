import pdi.segmentation as segmentation
import control.constant as const
import numpy as np
import cv2
import os

class Image():
    def __init__(self, name, img):
        self.name = os.path.basename(name).split(".")[0]
        self.original = img
        self.label = None
        self.matrix = None

def histogram(img):
    h, w = len(img), len(img[0])
    hist = np.zeros(256, dtype=int)

    for y in range(h):
        for x in range(w):
            hist[int(img[y][x])] += 1

    return hist

def otsu(img):
    # ret, th = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return segmentation.otsu(img)

def preprocessor(image):
    dsize = (const.IMAGE_SIZE[0], const.IMAGE_SIZE[1])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize)
    image = otsu(image)

    image = np.reshape(image, image.shape+(1,))
    image = np.reshape(image,(1,)+image.shape)

    return image