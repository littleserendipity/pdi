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