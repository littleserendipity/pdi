import control.constant as const
import os

''' Folder '''

# > project level
dn_DATA = "dataset"
# >> data level
dn_TEST = "test"
dn_TRAIN = "train"
# >>> only train level
dn_AUGMENTATION = "aug"
# >>> test and train level
dn_IMAGE = "image"
dn_LABEL = "label"

# > project level
dn_OUT = "out"
# >> out level
dn_TOLABEL = "tolabel"

# > src level
dn_ARCH = "arch"
dn_MODEL = "model"

''' File '''

# > src level
# >> model level
fn_CHECKPOINT = ""
fn_LOGGER = ""

''' Information '''

DATASET = ""
MODEL = ""
IMAGE_SIZE = (256,256)

VALIDATION = False
p_VALIDATION = 0.1

def setup(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.gpu else "-1"

    const.MODEL = args.arch
    const.DATASET = args.dataset
    const.VALIDATION = args.validation

    const.fn_CHECKPOINT = ("%s_%s_checkpoint.hdf5" % (const.MODEL, const.DATASET))
    const.fn_LOGGER = ("%s_%s_logger.log" % (const.MODEL, const.DATASET))