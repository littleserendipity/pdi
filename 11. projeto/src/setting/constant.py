''' Default '''
DATASET = "cracktile"
MODEL = "unet"
IMG_PROCESSING = "cracktile"

IMAGE_SIZE = (256,256)
p_VALIDATION = 0.1


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

dn_PDI = "pdi"
# >> PDI level
dn_PROCESSING = "processing"

dn_MODEL = "model"


''' File '''
# > src level
# >> model level
fn_CHECKPOINT = "checkpoint.hdf5"
fn_LOGGER = "logger.log"