from keras.callbacks import ModelCheckpoint, CSVLogger
from util import path, data, misc
import control.constant as const
import numpy as np
import importlib

class NeuralNetwork():
    def __init__(self): 
        self.arch = importlib.import_module("%s.%s" % (const.dn_ARCH, const.MODEL))

        self.fn_checkpoint = path.fn_checkpoint()
        self.fn_logger = path.fn_logger()

        self.dn_train_image = path.dn_train(const.dn_TRAIN_IMAGE)
        self.dn_aug_image = path.dn_aug(const.dn_TRAIN_IMAGE, mkdir=False)

        self.dn_train_label = path.dn_train(const.dn_TRAIN_LABEL)
        self.dn_aug_label = path.dn_aug(const.dn_TRAIN_LABEL, mkdir=False)

def train():
    nn = NeuralNetwork()

    images = data.fetch_from_path(nn.dn_train_image, nn.dn_aug_image)
    labels = data.fetch_from_path(nn.dn_train_label, nn.dn_aug_label)

    t_images, g_labels, v_images, v_labels = misc.random_split_dataset(images, labels, const.p_VALIDATION)
    epochs, steps_per_epoch, validation_steps = misc.epochs_and_steps(len(t_images), len(v_images))

    generator = data.train_generator(t_images, g_labels)
    validation_data = data.train_generator(v_images, v_labels)

    model = nn.arch.model()
    checkpoint = ModelCheckpoint(nn.fn_checkpoint, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    logger = CSVLogger(nn.fn_logger)

    print("\ntrain_images:\t\t%s | epochs:\t%s | steps_per_epoch:\t%s\n"
        "validation_images:\t%s | epochs:\t%s | validation_steps:\t%s\n" 
        % misc.str_center(len(t_images), epochs, steps_per_epoch, len(v_images), epochs, validation_steps))

    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps,
        validation_data=validation_data,
        use_multiprocessing=True,
        callbacks=[checkpoint, logger])

def test():
    nn = NeuralNetwork()



    print("test")