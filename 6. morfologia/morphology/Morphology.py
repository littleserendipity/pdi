import numpy as np
import Image as im
import copy

class Morphology(object):

    def logicalOperator(self, arr1, arr2, operator):
        name = arr1.name + "_" + operator + "_" + arr2.name
        img = im.Image(name=name)
        op = operator.lower()

        if (op == "or"):
            temp = np.add(arr1.arr, arr2.arr)
            temp[temp > 0] = 1
            img.setImg(temp)

        elif (op == "and"):
            img.setImg(np.multiply(arr1.arr, arr2.arr))

        elif (op == "xor"):
            temp = np.subtract(arr1.arr, arr2.arr)
            temp[temp < 0] = 1
            img.setImg(temp)

        elif (op == "nand"):
            temp = np.add(arr1.arr, arr2.arr)
            temp[temp == 0] = 1
            temp[temp == 2] = 0
            img.setImg(temp)

        return img

    def dilate(self, image, kernel=None, side=3):
        if (kernel is None):
            kernel = np.zeros((side,side))
        return self.filterMorph(image, kernel, np.prod, np.multiply)
    
    def erode(self, image, kernel=None, side=3):
        if (kernel is None):
            kernel = np.zeros((side,side))
        return self.filterMorph(image, kernel, np.sum, np.add)

    def filterMorph(self, image, kernel, externalFunction, internalFunction):
        im = copy.deepcopy(image)
        m, n = kernel.shape
        pad_h, pad_w = (m//2), (n//2)
        H, W = image.arr.shape

        img = np.ones((H + pad_h * 2, W + pad_w * 2)) * 128
        new_img = np.ones((H + pad_h * 2, W + pad_w * 2))
        img[pad_h:-pad_h, pad_w:-pad_w] = image.arr

        for i in range(pad_h, H+pad_h):
            for j in range(pad_w, W+pad_w):
                current_window = img[i-pad_h:i+m-pad_h, j-pad_w:j+n-pad_w]
                new_img[i,j] = externalFunction(internalFunction(current_window, kernel))

        new_img[new_img > 0] = 1
        im.setImg(new_img[pad_h:-pad_h,pad_w:-pad_w])
        return im