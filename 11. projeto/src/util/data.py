from glob import glob
from util import path
import setting.constant as const
import numpy as np
import cv2

def length_from_path(file_dir, *dirs):
    length_fetch = len(glob(path.join(file_dir, const.FILTER)))
    for x in dirs:
        length_fetch += len(glob(path.join(x, const.FILTER)))
    return length_fetch

def fetch_from_path(file_dir, *dirs):
    fetch = sorted(glob(path.join(file_dir, const.FILTER)))
    items = np.array([cv2.imread(item, 1) for item in fetch])
    shape = items[0].shape[:2]

    for x in dirs:
        fetch = sorted(glob(path.join(x, const.FILTER)))
        if (fetch):
            temp = np.array([cv2.resize(cv2.imread(item, 1), dsize=shape) for item in fetch])
            items = np.concatenate((items, temp))

    return items

def imshow(name, image):
    image = np.clip(image, 0, 255)
    cv2.imshow(name, np.uint8(image))
    cv2.waitKey(0)

def imwrite(file_name, image):
    image = np.clip(image, 0, 255)
    cv2.imwrite(file_name, np.uint8(image))