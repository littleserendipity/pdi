from util import path, misc
from glob import glob
from pdi import pdi
import numpy as np
import cv2

def train_prepare(images, labels):
    for (image, label) in zip(images, labels):
        (image, label) = pdi.preprocessor(image, label)
        yield misc.image_to_keras(image), misc.image_to_keras(label)

def test_prepare(images):
    for image in images:
        image, _ = pdi.preprocessor(image, None)
        yield misc.image_to_keras(image)

def fetch_from_path(file_dir, *dirs):
    fetch = sorted(glob(path.join(file_dir, "*[0-9].*")))
    items = np.array([cv2.imread(item, 1) for item in fetch])
    shape = items[0].shape[:2]

    for x in dirs:
        fetch = sorted(glob(path.join(x, "*[0-9].*")))
        if (fetch):
            temp = np.array([cv2.resize(cv2.imread(item, 1), dsize=shape) for item in fetch])
            items = np.concatenate((items, temp))

    return items

def save_predict(dir_save, arr_original, arr):
    for (i, image) in enumerate(arr):
        number = ("%0.3d" % (i+1))
        path_save = path.join(dir_save, mkdir=True)
        file_name = ("predict_%s.png" % (number))
        file_save = path.join(path_save, file_name)

        image = pdi.posprocessor(arr_original[i], misc.keras_to_image(image))
        imwrite(file_save, image)

def imshow(name, image):
    image = np.clip(image, 0, 255)
    cv2.imshow(name, np.uint8(image))
    cv2.waitKey(0)

def imwrite(file_name, image):
    image = np.clip(image, 0, 255)
    cv2.imwrite(file_name, np.uint8(image))