import Utils as utl
import Morphology as mp
import numpy as np
import copy


import matplotlib.pyplot as plt


class Representation(object):
    def __init__(self):
        self.m = mp.Morphology()
        self.data = utl.Data()

    def setValues(self, directions):
        self.background = 0
        self.mod = directions//4

        if (directions == 8):
            # self.directions = (-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)
            self.directions = (-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(-1,0),(-1,-1) ### (0,-1) changed
            self.weight = np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7]), 2)

        elif (directions == 4):
            self.directions = (-1,0),(0,1),(1,0),(0,-1)
            self.weight = np.tile(np.array([0, 1, 2, 3]), 2)

        return self.directions

    def chain(self, image, directions=8, norm=True):
        img = copy.deepcopy(image)
        img.clear(times=img.noise)
        around = self.setValues(directions)

        if (img.arr[0,0] > self.background):
            img.setImg(np.logical_not(img.arr))

        n_img = np.zeros(img.shapes)
        ind = np.argwhere(img.arr != self.background)[0]
        match = (ind[0],ind[1])
        indice = (0,1)
        chain = []

        try:
            while match:
                img.arr[match] = 0
                n_img[match] = 1
                chain.append(self.getCodChain(indice, self.mod))
                match, indice, around = self.mooreNeighbor(img.arr, match, around, self.background)
        except:
            chain_norm = self.joinArray(self.normalize(chain)) if norm else []
            img.setImg(n_img)
        return [img, self.joinArray(chain), chain_norm]

    def mooreNeighbor(self, arr, match, around, background):
        for t in range(len(around)):
            y, x = np.add(match, around[t])
            if (arr[y,x] != background):
                return [(y,x), around[t], (around[t-2:] + around[:t-2])]
        return False

    def getCodChain(self, indice, mult):
        if indice == (-1, -1):
            return 3
        elif indice == (-1, 0):
            return 1 * mult
        elif indice == (-1, 1):
            return 1
        elif indice == (0, 1):
            return 0 * mult
        elif indice == (1, 1):
            return 1
        elif indice == (1, 0):
            return 3 * mult
        elif indice == (1, -1):
            return 1
        elif indice == (0, -1):
            return 2 * mult

    def joinArray(self, arr):
        return (''.join(map(str, arr)))

    def normalize(self, arr):
        return self.normalizeByRotation(self.normalizeByPosition(arr))

    def normalizeByPosition(self, arr):
        arr_str = self.joinArray(arr)
        temp, temp_str = [0], arr_str

        while True:
            temp.append(0)
            temp_str = str(''.join(map(str, temp)))
            if (arr_str.find(temp_str) == -1):
                temp = np.delete(temp, -1)
                temp_str = str(''.join(map(str, temp)))
                break

        index = arr_str.find(temp_str)
        arr_f = np.concatenate([arr[index:], arr[:index]])
        return arr_f

    def normalizeByRotation(self, arr):
        try:
            result = []
            for x in range(len(arr)):
                count, cur, nex = -1, arr[x], arr[x+1]

                for j in range(len(self.weight)):
                    if ((cur == nex or count > -1) and nex == self.weight[j]):
                        result.append(count+1)
                        break
                    if (count > -1):
                        count += 1
                    if (cur == self.weight[j]):
                        count = 0
        except:
            return result

    def mpp(self, image, side=15):
        img = copy.deepcopy(image)
        m,n = img.shapes
        n_img_temp = np.zeros((m,n))
        n_img = np.zeros((m,n))

        for y in range(0, m, side):
            for x in range(0, n, side):
                matrix = img.arr[y:y+side, x:x+side]
                if (np.sum(matrix) > 0):
                    n_img[y:y+side, x:x+side] += 1
                    n_img_temp[y:y+side, x:x+side] += 1
                    n_img_temp[y:y+side, x:x+side] *= np.logical_not(matrix)

        img.setImg(n_img_temp)
        img.save(extension="step_line")

        img.setImg(n_img)
        n_img = np.zeros((m,n))
        
        img = self.m.floodFill(img)
        chain = list(map(int,list(self.chain(img, directions=4, norm=False)[1])))
        match = np.argwhere(img.arr == self.background)[0]

        img2 = self.m.floodFill(img, start=match)
        img.setImg(np.logical_or(img.arr, np.logical_not(img2.arr)))

        for y in range(0, m, side):
            for x in range(0, n, side):
                if (np.sum(img.arr[y:y+side, x:x+side]) == 0):
                    d_up_left = np.sum(img.arr[y-side:y, x-side:x])
                    left = np.sum(img.arr[y:y+side, x-side:x])  
                    d_bottom_left = np.sum(img.arr[y+side:y+(side*2), x-side:x])
                    
                    # up = np.sum(img.arr[y-side:y, x:x+side])
                    # bottom = np.sum(img.arr[y+side:y+(side*2), x:x+side])
                    
                    d_up_right = np.sum(img.arr[y-side:y, x+side:x+(side*2)])
                    right = np.sum(img.arr[y:y+side, x+side:x+(side*2)])
                    d_bottom_right = np.sum(img.arr[y+side:y+(side*2), x+side:x+(side*2)])

                    if (left > 0 and d_up_left > 0):
                        n_img[y, x+1] = 1
                    if (left > 0 and d_bottom_left > 0):
                        n_img[y+side-1, x+1] = 1
                    if (right > 0 and d_up_right > 0):
                        n_img[y, x+side] = 1
                    if (right > 0 and d_bottom_right > 0):
                        n_img[y+side-1, x+side] = 1

        for x in range(0, len(chain)):
            # n_img[match[0],match[1]] = 0

            if (n_img[match[0],match[1]] > 0):
                print("Line: ", match[0], match[1])

            if (chain[x] == 0):
                match[1] += 1
            elif (chain[x] == 1):
                match[0] -= 1
            elif (chain[x] == 2):
                match[1] -= 1
            elif (chain[x] == 3):
                match[0] += 1

        # plt.plot([225, 255, 285], [45, 45, 60], color="black")
        # plt.show()

        img.setImg(n_img)
        img.save(extension="step_points")

        return img

    def saveChain(self, name, extension, arr):
        self.data.saveVariable(name, extension, arr)