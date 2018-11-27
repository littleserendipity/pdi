from keras.preprocessing.image import ImageDataGenerator
from util import path, data, image as im
import control.constant as const
import numpy as np

def augmentation(n=1):
    batch_size = 1
    target_size = const.IMAGE_SIZE
    seed = int(np.random.rand(1)*100)

    train_path = path.data(const.DATASET, const.dn_TRAIN)

    image_folder = image_save_prefix = const.dn_IMAGE
    label_folder = label_save_prefix = const.dn_LABEL

    image_to_dir = path.dn_aug(const.dn_IMAGE)
    label_to_dir = path.dn_aug(const.dn_LABEL)
    
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
        if (i >= n-1): break

def tolabel():
    dir_save = path.out(const.dn_TOLABEL)
    images = data.fetch_from_path(dir_save)

    for (i, image) in enumerate(images):
        path_save = path.join(dir_save, "label", mkdir=True)
        file_name = ("%0.3d.png" % (i))
        file_save = path.join(path_save, file_name)

        img_pp, _ = im.preprocessor(image, None)
        im.imwrite(file_save, img_pp)