''' Default '''
DATASET = "cracktile"
IMG_PROCESSING = "cracktile"
MODEL = "unet"
IMAGE_SIZE = (256,256,1)

### Blue, Green, Red
BACKGROUND_COLOR = [224, 120, 110]
SEGMENTATION_COLOR = [30, 30, 200]

p_VALIDATION = 0.2
MIN_DELTA = 1e-6
PATIENCE = 6
### Monitor: loss, acc, val_loss, val_acc
MONITOR = "val_loss"

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
dn_NN = "nn"
# >> NN level
dn_ARCH = "arch"

dn_DIP = "dip"
# >> DIP level
dn_PROCESSING = "processing"

dn_MODEL = "model"

''' File '''
# > src level
# >> model level
fn_CHECKPOINT = "checkpoint.hdf5"
fn_LOGGER = "logger.log"
fn_SEGMENTATION = "result.txt"