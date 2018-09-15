from matplotlib import pyplot as plt
import numpy as np
import Utils as utl

class Histogram():
    def __init__(self):
        self.path = utl.Path()

    def getValues(self, arr, show=False):
        y_arr = np.zeros(256, dtype=int)
        for y in range(len(arr)):
            for x in range(len(arr[0])):
                y_arr[int(arr[y,x])] += 1
        if (show):
            plt.show()
        return y_arr

    def diff(self, original, result):
        y_arr = np.subtract(original, result)
        y_arr[y_arr < 0] = 0
        return y_arr

    def save(self, y_arr, name, extension="histogram", color="black"):
        name = self.path.getNameResult(name, extension)
        x_arr = [x for x in range(256)]

        plt.bar(x_arr, y_arr, width=1, color=color)
        plt.plot(x_arr, y_arr, color=color)
        plt.title("Histograma")
        plt.xlabel("Pixel")
        plt.ylabel("Frequência")
        plt.savefig(self.path.getPathSave(name))
        plt.close()