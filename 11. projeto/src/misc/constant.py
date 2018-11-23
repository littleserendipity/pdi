import misc.constant as const
import os

F_DATA = "data"
F_OUT = "out"
F_MODEL = "model"

F_TEST = "test"
F_TRAIN = "train"
F_TRAIN_IMAGE = "image"
F_TRAIN_LABEL = "label"

F_AUGMENTATION = "aug"

DATASET = ""
MODEL_CHECKPOINT = ""
TXT_REPORT = ""
IMAGE_SIZE = (256, 256)

def setup(dt_set, gpu):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if gpu else "-1"

    const.DATASET = dt_set
    const.MODEL_CHECKPOINT = ("unet_%s.hdf5" % dt_set)
    const.TXT_REPORT = ("unet_%s.txt" % dt_set)