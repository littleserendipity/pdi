from PIL import Image
import numpy
import math
import time
import os

def eightNeighbours(img, pointC, yy, xx):
	pointsToCheck = numpy.array([[yy, xx-1], [yy-1, xx-1], [yy-1, xx], [yy-1, xx+1], [yy, xx+1], [yy+1, xx+1], [yy+1, xx], [yy+1, xx-1]])
	#pointsToCheckZero = numpy.array([[yy, xx-1, 0], [yy-1, xx-1,0], [yy-1, xx,0], [yy-1, xx+1,0], [yy, xx+1,0], [yy+1, xx+1,0], [yy+1, xx,0], [yy+1, xx-1,0]])
	eightPoints = numpy.array([4,3,2,1,0,7,6,5])
	for i in range(8):
		if numpy.array_equal(pointC, pointsToCheck[i]):
			firstPoint = i
			break
	for i in range(8):
		#print(pointsToCheck[firstPoint])
		if(img[pointsToCheck[firstPoint, 0], pointsToCheck[firstPoint, 1]] == 255):
			bPoint = [pointsToCheck[firstPoint, 0], pointsToCheck[firstPoint, 1]]
			direction = eightPoints[firstPoint]
			if(i > 0):
				cPoint = [pointsToCheck[firstPoint-1, 0], pointsToCheck[firstPoint-1, 1]]
			else:
				cPoint = [pointsToCheck[7, 0], pointsToCheck[7, 1]]
			break
		if(firstPoint < 7):
			firstPoint+=1
		else:
			firstPoint = 0
	return bPoint, cPoint, direction

def getDirectionValue(direction):
	directions = numpy.array([[0, 1], [-1,1], [-1, 0], [-1,-1], [0, -1], [1, -1], [1, 0], [1,1]]) #0 1 2 3 4 5 6 7
	return directions[direction]


def chaincode(img):
	path = "Images"
	save_path = os.path.join(path, "result_chaincode", img)
	im = Image.open(os.path.join(path, img), "r")
	imageArray = numpy.array(im, dtype=numpy.uint8)
	imageArray = imageArray[:, :, 0]
	#print(imageArray[0])
	width, height = im.size
	newImageArray = numpy.zeros((height, width), dtype=numpy.uint8)
	#encontrar ponto mais acima e a esquerda
	found = False
	chainEight = []
	for yy in range(height):
		for xx in range(width):
			if(imageArray[yy, xx] == 255):
				bZero = numpy.array([yy, xx])
				cZero = numpy.array([yy, xx-1])
				found = True
				break
		if(found):
			break
	#find bOne
	bOne, cOne, direction = eightNeighbours(imageArray, cZero, bZero[0], bZero[1])
	chainEight.append(direction)
	b = bOne
	c = cOne
	while(True):
		nk, nkminusone, direction = eightNeighbours(imageArray, c, b[0], b[1])
		if(numpy.array_equal(b,bZero) and numpy.array_equal(nk, bOne)):
			break
		chainEight.append(direction)
		b = nk
		c = nkminusone
	chainEight = numpy.array(chainEight)
	#print(chainEight)
	numpy.savetxt("chain.txt", chainEight, fmt="%d")
	#create the new image
	point = bZero
	print("Chain Eight Size: " + str(chainEight.size))
	print("First Point = " + str(point))
	for i in range(chainEight.size):
		#print("Point Before: " + str(point))
		newImageArray[point[0], point[1]] = 255
		point += getDirectionValue(chainEight[i])
		#print("Point After: " + str(point))

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	Image.fromarray(newImageArray).save(save_path)


begin = time.time()
chaincode("Image_(1).bmp")
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")
