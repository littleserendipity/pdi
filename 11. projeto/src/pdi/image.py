import os
import numpy as np
import pdi.segmentation as segmentation

class Image():
    def __init__(self, name, img):
        self.name = os.path.basename(name).split(".")[0]
        self.original = img
        self.matrix = None

def histogram(arr):
    h, w = len(arr), len(arr[0])
    hist = np.zeros(256, dtype=int)

    for y in range(h):
        for x in range(w):
            hist[int(arr[y][x])] += 1

    return hist

def otsu(arr):
    return segmentation.otsu(arr)