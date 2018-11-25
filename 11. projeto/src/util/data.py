from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from util import path, misc
import control.constant as const
import util.image as im
import numpy as np
import cv2

def augmentation(batch=1):
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
        if (i >= batch-1): break

def train_prepare(images, labels):
    for (image, label) in zip(images, labels):
        (image, label) = im.preprocessor(image), im.preprocessor(label, label=True)
        yield (image, label)

def test_prepare(images):
    for image in images:
        image = im.preprocessor(image)
        yield image

def fetch_from_path(file_dir, *dirs, gen=True):
    read = lambda x: cv2.resize(cv2.imread(x, 1), dsize=const.IMAGE_SIZE)

    fetch = sorted(glob(path.join(file_dir, "*[0-9].*")))
    items = np.array([read(item) for item in fetch])

    try:
        for x in dirs:
            fetch = sorted(glob(path.join(x, "*[0-9].*")))
            temp = np.array([read(item) for item in fetch])
            items = np.concatenate((items, temp))
    except Exception as e:
        print(e)
        pass

    total = len(items)
    q = misc.round_up(total, 100) - total

    if (gen and q > 0):
        temp_batch = ImageDataGenerator().flow(x=items, batch_size=q)
        temp, batch_index = [], 0

        while (batch_index < q):
            data = temp_batch.next()
            temp.append(data[0])
            batch_index = batch_index + 1

        items = np.concatenate((items, np.asarray(temp, dtype=np.uint8)))
    return items

def save_predict(dir_save, arr_original, arr):
    for (i, image) in enumerate(arr):
        number = ("%0.3d" % (i+1))
        path_save = path.join(dir_save, str(number), mkdir=True)
        file_name = ("predict_%s.png" % (number))
        file_save = path.join(path_save, file_name)

        image = im.posprocessor(arr_original[i], image[:,:,0])

        ### sobreposição de resultado com original ###

        im.imwrite(file_save, image)