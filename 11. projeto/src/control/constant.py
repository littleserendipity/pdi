import control.constant as const
import os

''' Folder '''

# > project level
dn_DATA = "dataset"
# >> data level
dn_TEST = "test"
dn_TRAIN = "train"
# >>> train level
dn_TRAIN_IMAGE = "image"
dn_TRAIN_LABEL = "label"

# > project level
dn_OUT = "out"
# >> out level
dn_AUGMENTATION = "aug"

# > src level
dn_ARCH = "arch"
dn_MODEL = "model"

''' File '''

# > src level
# >> model level
fn_CHECKPOINT = ""
fn_LOGGER = ""

''' Information '''

MODEL = ""
DATASET = ""

IMAGE_SIZE = (256,256,1)
p_VALIDATION = 0.1
p_EPOCH = 0.02

def setup(dt_set, dt_model, gpu):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if gpu else "-1"

    const.MODEL, const.DATASET = dt_model, dt_set
    const.fn_CHECKPOINT = ("%s_%s_checkpoint.hdf5" % (dt_model, dt_set))
    const.fn_LOGGER = ("%s_%s_logger.log" % (dt_model, dt_set))