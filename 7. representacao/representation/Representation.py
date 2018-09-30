import Utils as utl
import Image as im
import numpy as np
import copy

class Representation(object):
    def __init__(self):
        self.data = utl.Data()

    def saveChain(self, name, extension, arr):
        self.data.saveVariable(name, extension, arr)

    def setValues(self, directions):
        if (directions == 8):
            # self.directions = (-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)
            self.directions = (-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(-1,0),(-1,-1) ### (0,-1) changed
            self.weight = np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7]), 2)

        elif (directions == 4):
            self.directions = (-1,0),(0,1),(1,0),(0,-1)
            self.weight = np.tile(np.array([0, 1, 2, 3]), 2)

        return self.directions

    def chain(self, image, bg=0, directions=8):
        img = copy.deepcopy(image)
        img.clear(times=img.noise)
        n_img = np.zeros(img.shapes)
        around = self.setValues(directions)

        ind = np.argwhere(img.arr != bg)[0]
        match = (ind[0],ind[1])
        indice = (0,1)
        chain = []

        try:
            while match:
                img.arr[match] = 0
                n_img[match] = 1
                chain.append(self.getCodChain(indice, len(around)//4))
                match, indice, around = self.mooreNeighbor(img.arr, match, around, bg)
        except:
            chain_norm = self.normalize(chain)
            img.setImg(n_img)
        return (img, self.joinArray(chain), self.joinArray(chain_norm))

    def mooreNeighbor(self, arr, match, around, bg):
        for t in range(len(around)):
            y, x = np.add(match, around[t])
            if (arr[y,x] != bg):
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