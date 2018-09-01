from PIL import Image
import numpy
import math
import time
import os

numpy.set_printoptions(threshold=numpy.nan)


def DFT(array1D, invert):
        M = len(array1D)
        multiplier = 1/(math.sqrt(M))
        #numReal = []
        #numImaginary = []
        numComplete = []
        for m in range(M):
                sumReal = 0
                sumImaginary = 0
                for u in range(M):
                        cosSenAtt = 2 * math.pi * ((m * u)/M)
                        cos = math.cos(cosSenAtt)
                        sin = math.sin(cosSenAtt)
                        if(invert):
                                sin = -sin
                        sumReal += array1D[u] * cos
                        sumImaginary += array1D[u] * -1j * sin
                sumReal *= multiplier
                sumImaginary *= multiplier
                #numReal.append(sumReal)
                #numImaginary.append(sumImaginary)
                if(invert):
                	numComplete.append(int(round(numpy.abs(sumReal + sumImaginary))))
                else:
                	numComplete.append(sumReal + sumImaginary)

        #print(array1D)
        #print(numReal)
        #print(numImaginary)
        #print(numComplete)

        return numComplete

def DFT2D(array2D, invert):
        M = len(array2D)
        N = len(array2D[0])
        firstMatrix = []
        secondMatrix = []

        for row in range(M):
                tempRow = []
                for col in range(N):
                        tempRow.append(array2D[row][col])
                newRow = DFT(tempRow, invert)
                firstMatrix.append(newRow)
        for col in range(N):
                tempCol = []
                for row in range(M):
                        tempCol.append(firstMatrix[row][col])
                newCol = DFT(tempCol, invert)
                secondMatrix.append(newCol)
        #print(secondMatrix)
        #print("Inverted: ")
        matrixToNumpy = numpy.matrix(secondMatrix)
        #print(matrixToNumpy.transpose())
        return secondMatrix


def filter(img):
        path = "Images"
        #save_path = os.path.join(path, "result_frequencia", "resultado_" + img)
        im = Image.open(os.path.join(path, img), "r")
        imageArray = numpy.array(im, dtype=numpy.uint8)
        width, height = im.size
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #Image.fromarray(newImageArray).save(save_path)

testArray = [1, 3, 5, 7, 9, 8, 6, 4, 2, 0]
testMatrix = [[5, 3, 0, 2], [1, 7, 8, 3], [4, 2, 2, 2], [8, 5, 2, 1]]
numpyTestArray = numpy.array(testArray)
numpyTestMatrix = numpy.matrix(testMatrix)
#DFT(testArray)
fourierTest1D = numpy.fft.fft(numpyTestArray)
fourierTest2D = numpy.fft.fft2(numpyTestMatrix)
print("Original Array: ")
print(testArray)
print("1D DFT Test: ")
test1D = DFT(testArray, False)
print(test1D)
print("1D DFT Invert Test")
test1DInvert = DFT(test1D, True)
print(test1DInvert)

print("Original Matrix: ")
print(testMatrix)
print("2D DFT Test: ")
test2D = DFT2D(testMatrix, False)
print(test2D)
print("2D DFT Invert Test: ")
test2DInvert = DFT2D(test2D, True)
print(test2DInvert)

begin = time.time()
#filter("Agucar_(1).jpg")
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")
