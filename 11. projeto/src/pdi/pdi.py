import setting.constant as const
import importlib
import cv2

def preprocessor(image, label=None):
    pp = importlib.import_module("%s.%s.%s" % (const.dn_PDI, const.dn_PROCESSING, const.IMG_PROCESSING))

    image = cv2.resize(image, dsize=const.IMAGE_SIZE)
    image = pp.image_preprocessor(image)

    if (label is not None):
        label = cv2.resize(label, dsize=const.IMAGE_SIZE)
        label = pp.label_preprocessor(label)
    
    return (image, label)
    
def posprocessor(original, image):
    pp = importlib.import_module("%s.%s.%s" % (const.dn_PDI, const.dn_PROCESSING, const.IMG_PROCESSING))

    image = cv2.resize(image, original.shape[:2][::-1])

    image = pp.posprocessor(image)
    return image