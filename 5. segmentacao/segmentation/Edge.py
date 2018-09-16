import numpy as np
import Image as im

class Edge(object):
    def __init__(self):
        self.kernel = np.array([
            [0, 0,   1, 0, 0],
            [0, 1,   2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1,   2, 1, 0],
            [0, 0,   1, 0, 0],
        ])
        self.sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ])
        self.sobel_y = np.array([
            [ 1,  2,  1],
            [ 0,  0,  0],
            [-1, -2, -1],
        ])

    def laplaceofGaussian(self, image):
        self.img = image
        self.img.clear(times=self.img.noise)

        G_x = self.img.convolve(self.sobel_x)
        G_y = self.img.convolve(self.sobel_y)

        G = np.power(np.add(np.square(G_x), np.square(G_y)), 0.5)
        G = (G>32) * G

        (M,N) = self.img.shapes
        img = np.zeros((M,N))
        temp = np.zeros((M+2, N+2))
        temp[1:-1,1:-1] = self.img.convolve(self.kernel)
        around = (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)

        for i in range(1,M+1):
            for j in range(1,N+1):
                if (temp[i,j] < 0):
                    for x,y in around:
                        if (temp[i+x,j+y] > 0):
                            img[i-1,j-1] = 1

        self.img.setImg(np.logical_not(np.logical_and(img, G)))
        return self.img