from keras.callbacks import ModelCheckpoint, CSVLogger
from util import path, data, misc
import control.constant as const
import numpy as np
import importlib

def train():
    arch = importlib.import_module("%s.%s" % (const.dn_ARCH, const.MODEL))

    images = data.fetch_from_path(path.dn_train(const.dn_TRAIN_IMAGE))
    labels = data.fetch_from_path(path.dn_train(const.dn_TRAIN_LABEL))

    t_images, g_labels, v_images, v_labels = misc.random_split_dataset(images, labels, const.p_VALIDATION)
    epochs, steps_per_epoch, validation_steps = misc.epochs_and_steps(len(t_images), len(v_images), const.p_EPOCH)

    generator = data.train_generator(t_images, g_labels)
    validation_data = data.train_generator(v_images, v_labels)

    model = arch.model()
    checkpoint = ModelCheckpoint(path.fn_checkpoint(), monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    logger = CSVLogger(path.fn_logger())

    print("\nt_images: %s\tepochs: %s\tsteps_per_epoch: %s" % (len(t_images), epochs, steps_per_epoch))
    print("v_images: %s\tepochs: %s\tsteps_per_epoch: %s\n" % (len(v_images), epochs, validation_steps))

    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps,
        validation_data=validation_data,
        use_multiprocessing=True,
        callbacks=[checkpoint, logger])

def test():
    print("test")