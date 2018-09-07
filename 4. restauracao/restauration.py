from PIL import Image as PIL
from matplotlib import pyplot as plt
from random import gauss
import numpy as np
import os

class Noise():
    def __init__(self, uni_p=0.15, bi_p=0.10, g_mean=10, g_sigma=30, g_n=100):
        self.uni_p = uni_p
        self.bi_p = bi_p
        self.gauss_mean = g_mean
        self.gauss_sigma = g_sigma
        self.gauss_n = g_n
        self.gauss = None
        self.gaussianGenerate()

    def makeNoise(self, img):
        matrix_uni = self.unipolar(img.shapes)
        matrix_bi = self.bipolar(img.shapes)
        matrix_gauss = self.gaussian(img.shapes)

        img_uni = Image(np.add(img.arr, matrix_uni), img.name)
        img_bi = Image(np.add(img.arr, matrix_bi), img.name)
        img_gauss = Image(np.add(img.arr, matrix_gauss), img.name)

        img_uni.save("(noise)_unipolar")
        img_bi.save("(noise)_bipolar")
        img_gauss.save("(noise)_gauss")

    def unipolar(self, shapes):
        return np.random.choice(a=[-255,0], size=(shapes[0], shapes[1]), p=[self.uni_p, (1-self.uni_p)])

    def bipolar(self, shapes):
        return np.random.choice([-255,0,255], size=(shapes[0], shapes[1]), p=(0.05, 0.9, 0.05))

    def gaussian(self, shapes):
        return np.random.choice(self.gauss, size=(shapes[0], shapes[1]))
    
    def gaussianGenerate(self):
        self.gauss = [gauss(self.gauss_mean, self.gauss_sigma) for i in range(self.gauss_n)]

class Image():
    def __init__(self, img=None, name=None):
        self.path = "images"
        self.path_save = None
        self.name = name
        self.arr = None
        self.shapes = None
        self.type = None
        self.load(img, name)

    def load(self, img, name="image"):
        if (isinstance(img, str)):
            self.name, self.type = img.split(".")
            self.arr = np.array(PIL.open(os.path.join(self.path, img)).convert('L'))
        else:
            self.name, self.type = name, "jpg"
            self.arr = img

        self.path_save = os.path.join(self.path, self.name)
        self.shapes = self.arr.shape
    
    def show(self, extension=None, mode="Greys_r"):
        plt.imshow(self.arr, cmap=mode, vmin=0, vmax=255)
        plt.show()

    def save(self, extension=None, mode="Greys_r"):
        plt.imsave(self.getPathToSave(extension), self.arr, cmap=mode, vmin=0, vmax=255)

    def getPathToSave(self, extension="result"):
        os.makedirs(self.path_save, exist_ok=True)
        save_file = self.name + "_" + extension + "." + self.type
        return os.path.join(self.path_save, save_file)

    def convolve(self, mask):
        mask = np.asarray(mask, dtype=float)
        m, n = mask.shape

        pad_h, pad_w = (m//2), (n//2)
        H, W, P = self.shapes

        img = np.ones((H + pad_h * 2, W + pad_w * 2, P)) * 128
        new_img = np.ones((H + pad_h * 2, W + pad_w * 2))
        img[pad_h:-pad_h, pad_w:-pad_w] = self.arr

        for i in range(pad_h, H+pad_h):
            for j in range(pad_w, W+pad_w):
                new_img[i,j] = sum(sum(np.mean(img[i-pad_h:i+m-pad_h, j-pad_w:j+n-pad_w]) * mask))

        return new_img[pad_h:-pad_h,pad_w:-pad_w]

    def histogram(self, extension="histogram", color="black", show=False, save=False):
        y_arr = np.zeros(256)
        x_arr = [x for x in range(256)]

        for y in range(self.shapes[0]):
            for x in range(self.shapes[1]):
                y_arr[self.arr[y,x]] += 1

        plt.bar(x_arr, y_arr, width=1, color=color)
        plt.plot(x_arr, y_arr, color=color)
        plt.title("Histograma")
        plt.xlabel("Pixel")
        plt.ylabel("Frequência")
        
        if (save):
            plt.savefig(self.getPathToSave(extension))
        elif (show):
            plt.show()
        return y_arr

### Main ###
def main():

    ### 1ª questão
    Noise().makeNoise(Image("image_(1).jpg"))

    ### 2ª questão

        # "image_(2).jpg"
        # "image_(3).jpg"
        # "image_(4).jpg"

    # img = Image("name")
    # curr_histogram = img.histogram(save=True)
        
if __name__ == "__main__":
    main()