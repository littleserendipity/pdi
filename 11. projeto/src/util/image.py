import control.constant as const
import util.misc as misc
import numpy as np
import cv2

def preprocessor(image, label=None):
    image = cv2.resize(image, dsize=const.IMAGE_SIZE)

    image = equalize_light(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    black_level = back_in_black(image)

    image = gauss_filter(image, (5,5))
    image = light(image, bright=-30, contrast=-30)
    
    if not black_level:
        image = cv2.bitwise_not(image)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(image, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    image = np.subtract(image, mask)
    image = cv2.bitwise_not(image)

    image = median_filter(image, 3)
    image = threshold(image, clip=3)

    if (label is not None):
        label = cv2.resize(label, dsize=const.IMAGE_SIZE)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = threshold(label, min_limit=254)

    return (image, label)
    
def posprocessor(original, result):
    result = cv2.resize(result, original.shape[:2][::-1])
    return result

def light(image, bright, contrast):
    bright = bright * 1.2
    contrast = contrast * 2
    image = image * ((contrast/127)+1) - contrast + bright
    image = misc.clip(image, 0, 255)
    return np.uint8(image)

def threshold(image, min_limit=None, max_limit=255, clip=0):
    if min_limit is None:
        min_limit = int(np.mean(image) - clip)

    _, image = cv2.threshold(image, min_limit, max_limit, cv2.THRESH_BINARY)
    return np.uint8(image)

def gauss_filter(image, kernel=(3,3), iterations=1):
    for _ in range(iterations):
        image = cv2.GaussianBlur(image, kernel, 0)
    return np.uint8(image)

def median_filter(image, kernel=3, iterations=1):
    for _ in range(iterations):
        image = cv2.medianBlur(image, kernel, 0)
    return np.uint8(image)

# def edges(image, threshold1=250, threshold2=350, kernel=3):
#     image = cv2.Canny(image, threshold1, threshold2, kernel)
#     image = cv2.bitwise_not(image)
#     return np.uint8(image)

# def equalize_hist(image):
#     image = cv2.equalizeHist(image)
#     return np.uint8(image)

def equalize_light(image, limit=3, grid=(7,7)):
    try:
        gray = True
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    except:
        gray = False
        pass

    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image)

def back_in_black(image):
    image = light(image.copy(), bright=60, contrast=60)
    black_level = 0

    for x in range(6):
        bi = threshold(image, clip=x)
        if (bi==0).sum() > (bi==255).sum():
            black_level += 1

    return (black_level > 3)

def keras_to_image(image):
    image = image[:,:,0]
    image = misc.clip(image, 0, 1)
    return np.multiply(image, 255)

def image_to_keras(image):
    image = np.reshape(image, image.shape+(1,))
    image = np.reshape(image,(1,)+image.shape)
    image = misc.clip(image, 0, 255)
    return np.divide(image, 255)

def imshow(name, image):
    image = misc.clip(image, 0, 255)
    cv2.imshow(name, np.uint8(image))
    cv2.waitKey(0)

def imwrite(file_name, image):
    image = misc.clip(image, 0, 255)
    cv2.imwrite(file_name, np.uint8(image))