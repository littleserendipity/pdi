from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from control import generator as gen, constant as const
from util import path, data, misc
import importlib

class NeuralNetwork():
    def __init__(self, test=False): 
        self.arch = importlib.import_module("%s.%s" % (const.dn_ARCH, const.MODEL))
        self.model = self.arch.model()
        self.loaded = False

        self.fn_checkpoint = path.fn_checkpoint()
        self.fn_logger = path.fn_logger()

        self.dn_IMAGE = path.dn_train(const.dn_IMAGE)
        self.dn_aug_image = path.dn_aug(const.dn_IMAGE, mkdir=False)

        self.dn_LABEL = path.dn_train(const.dn_LABEL)
        self.dn_aug_label = path.dn_aug(const.dn_LABEL, mkdir=False)

        self.dn_test = path.dn_test()
        self.dn_test_out = path.dn_test(out_dir=test, mkdir=False)

        if (path.exist(self.fn_checkpoint)):
            print("%s loaded" % self.fn_checkpoint)
            self.model.load_weights(self.fn_checkpoint)
            self.loaded = True

def train():
    nn = NeuralNetwork()

    images = data.fetch_from_path(nn.dn_IMAGE, nn.dn_aug_image)
    labels = data.fetch_from_path(nn.dn_LABEL, nn.dn_aug_label)

    total = len(images)
    q = misc.round_up(total, 100) - total

    if (q > 0):
        del images, labels
        print("\nDataset augmentation (90 increase) is necessary (only once)")
        gen.augmentation(q)

        images = data.fetch_from_path(nn.dn_IMAGE, nn.dn_aug_image)
        labels = data.fetch_from_path(nn.dn_LABEL, nn.dn_aug_label)
    
    if const.VALIDATION:
        images, labels, v_images, v_labels = misc.random_split_dataset(images, labels, const.p_VALIDATION)
        validation_data = data.train_prepare(v_images, v_labels)
        early_monitor = "val_loss" 
    else:
        v_images = []
        validation_data = None
        early_monitor = "loss" 

    generator = data.train_prepare(images, labels)
    epochs, steps_per_epoch, validation_steps = misc.epochs_and_steps(len(images), len(v_images))

    print("\nepochs: %s\ntrain size:\t\t%s |\tsteps_per_epoch: \t%s\nvalidation size:\t%s |\tvalidation_steps:\t%s\n" 
        % misc.str_center(epochs, len(images), steps_per_epoch, len(v_images), validation_steps))

    checkpoint = ModelCheckpoint(nn.fn_checkpoint, monitor='loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor=early_monitor, min_delta=1e-4, patience=5, verbose=1, restore_best_weights=True, mode='auto')
    logger = CSVLogger(nn.fn_logger)

    nn.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps,
        validation_data=validation_data,
        use_multiprocessing=True,
        callbacks=[checkpoint, early_stopping, logger])

def test():
    nn = NeuralNetwork(test=True)

    if (nn.loaded):
        images = data.fetch_from_path(nn.dn_test)
        generator = data.test_prepare(images)

        results = nn.model.predict_generator(generator, len(images), verbose=1)
        data.save_predict(nn.dn_test_out, images, results)
    else:
        print("\n>> Model not found (%s)\n" % nn.fn_checkpoint)