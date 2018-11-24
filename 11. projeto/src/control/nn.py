from keras.callbacks import ModelCheckpoint
from util import path, data, misc
import control.constant as const
import numpy as np
import importlib

def train():
    arch = importlib.import_module("%s.%s" % (const.dn_ARCH, const.MODEL))

    images, labels = data.load_dataset()
    g_images, g_labels, v_images, v_labels = misc.random_split_dataset(images, labels, const.p_VALIDATION)

    epochs, steps_per_epoch = misc.epochs_and_steps(len(g_images), const.p_EPOCH)
    _, validation_steps = misc.epochs_and_steps(len(v_images), const.p_EPOCH)

    generator = data.train_generator(g_images, g_labels)
    validation_data = data.train_generator(v_images, v_labels)

    model = arch.model(path.fn_checkpoint())
    model_checkpoint = ModelCheckpoint(path.fn_checkpoint(), monitor='loss', verbose=1, save_best_only=True)

    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps,
        validation_data=validation_data,
        use_multiprocessing=True,
        callbacks=[model_checkpoint])

def test():
    print("test")