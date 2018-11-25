import numpy as np
import cv2

def preprocessor(image, label=False):
    image = np.divide(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255)

    if label:
        image = otsu(image)
    else:
        # image = otsu(image)
        image[image <= 0.5] = 0
        image[image > 0.5] = 1

        ### arquitetura de PDI para as rachaduras ###

    return adjust_keras(image)

def posprocessor(image, result):
    return otsu(result)

def otsu(arr):
    # ret, th = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    image = get_im(arr)
    hist = histogram(image)
    total = (len(image) * len(image[0]))

    current_max, threshold = 0, 0
    sumT, sumF, sumB = 0, 0, 0

    weightB, weightF = 0, 0
    varBetween, meanB, meanF = 0, 0, 0

    for i in range(0, 256):
        sumT += (i * hist[i])

    for i in range(0, 256):
        weightB += hist[i]
        weightF = total - weightB
        if (weightF <= 0): break
        if (weightB <= 0): weightB = 1

        sumB += (i * hist[i])
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = (weightB * weightF)
        varBetween *= (meanB-meanF) * (meanB-meanF)

        if (varBetween > current_max):
            current_max = varBetween
            threshold = i

    image[image <= threshold] = 0
    image[image > threshold] = 1
    return image

def histogram(image):
    h, w = len(image), len(image[0])
    hist = np.zeros(256, dtype=int)
    for y in range(h):
        for x in range(w):
            hist[int(image[y][x])] += 1
    return hist

def imshow(name, image):
    cv2.imshow(name, np.array(get_im(image), dtype=np.uint8))
    cv2.waitKey(0)

def imwrite(file_name, image):
    cv2.imwrite(file_name, np.array(get_im(image), dtype=np.uint8))

def get_im(image):
    return np.multiply(image, 255) if (np.max(image) <= 1) else image

def adjust_keras(image):
    image = np.reshape(image, image.shape+(1,))
    image = np.reshape(image,(1,)+image.shape)
    return image