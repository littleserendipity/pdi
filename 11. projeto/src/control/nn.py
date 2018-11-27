from keras.callbacks import ModelCheckpoint, CSVLogger
from util import path, data, misc
import control.constant as const
import numpy as np
import importlib

class NeuralNetwork():
    def __init__(self, test=False): 
        self.arch = importlib.import_module("%s.%s" % (const.dn_ARCH, const.MODEL))

        self.fn_checkpoint = path.fn_checkpoint()
        self.fn_logger = path.fn_logger()

        self.dn_IMAGE = path.dn_train(const.dn_IMAGE)
        self.dn_aug_image = path.dn_aug(const.dn_IMAGE, mkdir=False)

        self.dn_LABEL = path.dn_train(const.dn_LABEL)
        self.dn_aug_label = path.dn_aug(const.dn_LABEL, mkdir=False)

        self.dn_test = path.dn_test()
        self.dn_test_out = path.dn_test(out_dir=test, mkdir=False)

def train():
    nn = NeuralNetwork()

    images = data.fetch_from_path(nn.dn_IMAGE, nn.dn_aug_image)
    labels = data.fetch_from_path(nn.dn_LABEL, nn.dn_aug_label)

    t_images, g_labels, v_images, v_labels = misc.random_split_dataset(images, labels, const.p_VALIDATION)
    epochs, steps_per_epoch, validation_steps = misc.epochs_and_steps(len(t_images), len(v_images))

    generator = data.train_prepare(t_images, g_labels)
    validation_data = data.train_prepare(v_images, v_labels)

    model = nn.arch.model()
    checkpoint = ModelCheckpoint(nn.fn_checkpoint, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    logger = CSVLogger(nn.fn_logger)

    print("\nepochs:%s\ntrain size:\t\t%s |\tsteps_per_epoch: \t%s\nvalidation size:\t%s |\tvalidation_steps:\t%s\n" 
        % misc.str_center(epochs, len(t_images), steps_per_epoch, len(v_images), validation_steps))

    model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps,
        validation_data=validation_data,
        use_multiprocessing=True,
        callbacks=[checkpoint, logger])

def test():
    nn = NeuralNetwork(test=True)

    if (path.exist(nn.fn_checkpoint)):
        model = nn.arch.model(nn.fn_checkpoint)

        images = data.fetch_from_path(nn.dn_test, gen=False)
        generator = data.test_prepare(images)

        results = model.predict_generator(generator, len(images), verbose=1)
        data.save_predict(nn.dn_test_out, images, results)
    else:
        print("\n>> Model not found (%s)" % nn.fn_checkpoint)