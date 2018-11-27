from glob import glob
from util import path
import util.image as im
import numpy as np
import cv2

def train_prepare(images, labels):
    for (image, label) in zip(images, labels):
        (image, label) = im.preprocessor(image, label)
        yield im.image_to_keras(image), im.image_to_keras(label)

def test_prepare(images):
    for image in images:
        image, _ = im.preprocessor(image, None)
        yield im.image_to_keras(image)

def fetch_from_path(file_dir, *dirs):
    fetch = sorted(glob(path.join(file_dir, "*[0-9].*")))
    items = np.array([cv2.imread(item, 1) for item in fetch])

    for x in dirs:
        fetch = sorted(glob(path.join(x, "*[0-9].*")))
        if (fetch):
            temp = np.array([cv2.imread(item, 1) for item in fetch])
            items = np.concatenate((items, temp))

    return items

def save_predict(dir_save, arr_original, arr):
    for (i, image) in enumerate(arr):
        number = ("%0.3d" % (i+1))
        path_save = path.join(dir_save, mkdir=True)
        file_name = ("predict_%s.png" % (number))
        file_save = path.join(path_save, file_name)

        image = im.posprocessor(arr_original[i], im.keras_to_image(image))

        ### sobreposição de resultado com original ###

        im.imwrite(file_save, image)