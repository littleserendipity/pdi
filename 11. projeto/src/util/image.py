from control.constant import IMAGE_SIZE
import numpy as np
import cv2

def preprocessor(image, label):
    return image_processing(image), label_processing(label)
    
def image_processing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMAGE_SIZE)
    image = otsu(image)
    return adjust_keras(image)

def label_processing(label):
    label[label<255] = 0
    return adjust_keras(label)

def adjust_keras(image):
    image = np.reshape(image, image.shape+(1,))
    image = np.reshape(image,(1,)+image.shape)
    return image

def otsu(image):
    # ret, th = cv2.threshold(gray, thresh, maxValue, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if (np.max(image) <= 1): 
        image = np.multiply(image, 255)

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
    return np.array(image, dtype=int)

def histogram(image):
    h, w = len(image), len(image[0])
    hist = np.zeros(256, dtype=int)
    for y in range(h):
        for x in range(w):
            hist[int(image[y][x])] += 1
    return hist