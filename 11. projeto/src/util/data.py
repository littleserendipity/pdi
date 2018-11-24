from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import control.constant as const
import util.path as path
import pdi.image as im
import numpy as np
import cv2

def augmentation(batch=1):
    batch_size = 1
    target_size = const.IMAGE_SIZE
    seed = int(np.random.rand(1)*100)

    train_path = path.data(const.DATASET, const.dn_TRAIN)

    image_folder = image_save_prefix = const.dn_TRAIN_IMAGE
    label_folder = label_save_prefix = const.dn_TRAIN_LABEL

    image_to_dir = path.out("%s_%s" % (const.dn_AUGMENTATION, const.DATASET), const.dn_TRAIN_IMAGE)
    label_to_dir = path.out("%s_%s" % (const.dn_AUGMENTATION, const.DATASET), const.dn_TRAIN_LABEL)
    
    image_gen = label_gen = ImageDataGenerator(
        rotation_range=90, 
        width_shift_range=0.025,
        height_shift_range=0.025, 
        channel_shift_range=0.025,
        shear_range=0.025,
        zoom_range=0.025,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode="reflect")

    image_batch = image_gen.flow_from_directory(
        directory = train_path,
        classes = [image_folder],
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = image_to_dir,
        save_prefix = image_save_prefix,
        seed = seed)

    label_batch = label_gen.flow_from_directory(
        directory = train_path,
        classes = [label_folder],
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = label_to_dir,
        save_prefix = label_save_prefix,
        seed = seed)

    for i, (_,_)  in enumerate(zip(image_batch, label_batch)):
        if (i >= batch-1): break

def train_generator(images, labels):
    for (image, label) in zip(images, labels):
        (image, label) = im.preprocessor(image, label)
        yield (image, label)

def fetch_from_path(file_dir):
    fetch = sorted(glob(path.join(file_dir, "*[0-9].*")))
    items = np.array([cv2.imread(item) for item in fetch])
    return items