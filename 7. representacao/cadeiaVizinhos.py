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

def normalizeChain(chain):
	chainNormalized = []
	for i in range(chain.size-1):
		if(i == 0):
			valueNormalized = (chain[i] - chain[chain.size-1]) % 8
		else:
			valueNormalized = (chain[i+1] - chain[i]) % 8
		chainNormalized.append(valueNormalized)
	return numpy.array(chainNormalized)

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
	chainNormalized = normalizeChain(chainEight)
	#print(chainEight)
	numpy.savetxt("chain.txt", chainEight, fmt="%d")
	numpy.savetxt("chainNormalized.txt", chainNormalized, fmt="%d")
	#create the new image
	point = bZero
	print("Chain Normalized Size: " + str(chainNormalized.size))
	print("First Point = " + str(point))
	for i in range(chainEight.size):
		#print("Point Before: " + str(point))
		if(point[0] < height and point[1] < width and point[0] > 0 and point[1] > 0):
			newImageArray[point[0], point[1]] = 255
		point += getDirectionValue(chainEight[i])
		#print("Point After: " + str(point))
	#return newImageArray
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	Image.fromarray(newImageArray).save(save_path)


def skeleton(img):
	path = "Images"
	save_path = os.path.join(path, "result_skeleton", img)
	save_path2 = os.path.join(path, "result_skeleton", "skeleton_" + img)
	save_path3 = os.path.join(path, "result_skeleton", "skeletonInverted_" + img)
	im = Image.open(os.path.join(path, img), "r")
	imageArray = numpy.array(im, dtype=numpy.uint8)
	print(imageArray[200, 200])
	#imageArray = imageArray[:1, :1]
	#print(imageArray)
	imageArrayWithSkeleton = numpy.copy(imageArray)
	print(imageArrayWithSkeleton[200, 200])
	#print(imageArray[0])
	width, height = im.size
	skeletonArray = numpy.full((height, width), 1, dtype=numpy.uint8)
	skeletonArrayInverted = numpy.full((height, width), 0, dtype=numpy.uint8)
	numIterations = 1
	while(True):
		# first step
		for yy in range(height):
			for xx in range(width):
				if(imageArray[yy, xx] > 0):
					# p9 p2 p3
					# p8 p1 p4
					# p7 p6 p5
					# imageArray[yy,xx] is p1
					# those are p2, p3, p4, p5, p6, p7, p8, p9, in this order
					pointsToCheck = numpy.array([[yy-1, xx], [yy-1, xx+1], [yy, xx+1], [yy+1, xx+1], [yy+1, xx], [yy+1, xx-1], [yy, xx-1], [yy-1, xx-1]])
					nPone = 0 # N(p1)
					tPone = 0 # T(p1)
					for i in range(8):
						# T(p1) = somatorio[(1 - Ti)*(Ti+1 mod 8)]
						ti = 0
						tiPlus = 0
						if(imageArray[pointsToCheck[i, 0], pointsToCheck[i, 1]] > 0):
							nPone +=1
							ti = 1
						nextI = (i+1) % 8
						if(imageArray[pointsToCheck[nextI, 0], pointsToCheck[nextI, 1]] > 0):
							tiPlus = 1
						tPone += (1 - ti) * (tiPlus % 8)
					fourSix =  imageArray[pointsToCheck[2, 0], pointsToCheck[2, 1]] * imageArray[pointsToCheck[4, 0], pointsToCheck[4, 1]]
					pTwoFourSix = imageArray[pointsToCheck[0, 0], pointsToCheck[0, 1]] * fourSix
					pFourSixEight = fourSix * imageArray[pointsToCheck[6, 0], pointsToCheck[6, 1]]
					#if(nPone >= 2 and nPone <= 6):
					#	print("tPone = " + str(tPone))
					#	print("pTwoFourSix = " + str(pTwoFourSix))
					#	print("pFourSixEight = " + str(pFourSixEight))
					#	print("nPone = " + str(nPone))
					if(nPone >= 2 and nPone <=6 and tPone == 1 and pTwoFourSix == 0 and pFourSixEight == 0):
					#	print("Entrou primeiro if!")
						skeletonArray[yy, xx] = 0
						skeletonArrayInverted[yy, xx] = 255
						imageArrayWithSkeleton[yy, xx] = 0
		# second step
		
		imageArray = numpy.copy(imageArrayWithSkeleton)
		for yy in range(height):
			for xx in range(width):
				if(imageArray[yy, xx] > 0):
					pointsToCheck = numpy.array([[yy-1, xx], [yy-1, xx+1], [yy, xx+1], [yy+1, xx+1], [yy+1, xx], [yy+1, xx-1], [yy, xx-1], [yy-1, xx-1]])
					nPone = 0 # N(p1)
					tPone = 0 # T(p1)
					for i in range(8):
						# T(p1) = somatorio[(1 - Ti)*(Ti+1 mod 8)]
						ti = 0
						tiPlus = 0
						if(imageArray[pointsToCheck[i, 0], pointsToCheck[i, 1]] > 0):
							nPone +=1
							ti = 1
						nextI = (i+1) % 8
						if(imageArray[pointsToCheck[nextI, 0], pointsToCheck[nextI, 1]] > 0):
							tiPlus = 1
						tPone += (1 - ti) * (tiPlus % 8)
					twoEight =  imageArray[pointsToCheck[0, 0], pointsToCheck[0, 1]] * imageArray[pointsToCheck[6, 0], pointsToCheck[6, 1]]
					pTwoFourEight = twoEight * imageArray[pointsToCheck[2, 0], pointsToCheck[2, 1]]
					pTwoSixEight = twoEight * imageArray[pointsToCheck[4, 0], pointsToCheck[4, 1]]
					if(nPone >= 2 and nPone <=6 and tPone == 1 and pTwoFourEight == 0 and pTwoSixEight == 0):
						#print("Entrou segundo if!")
						skeletonArray[yy, xx] = 0
						skeletonArrayInverted[yy, xx] = 255
						#print("Image array [yy, xx] = " + str(imageArray[yy, xx]))
						imageArrayWithSkeleton[yy, xx] = 0
						#print("Image array with skeleton [yy, xx] = " + str(imageArrayWithSkeleton[yy,xx]))
						#print("Image array again [yy, xx] = " +str(imageArray[yy, xx]))
		#print("Sao iguais = " + str(numpy.array_equal(imageArray, imageArrayWithSkeleton)))
		if(numpy.array_equal(imageArray, imageArrayWithSkeleton)):
			print("Sao iguais")
			break
		imageArray = numpy.copy(imageArrayWithSkeleton)
		numIterations += 1
		#print("Ué, chegou aqui")
	print("numIterations = " + str(numIterations))
	imageArray[imageArray > 0] = 255
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	Image.fromarray(imageArray).save(save_path)
	skeletonArray[skeletonArray > 0] = 255
	#print(skeletonArray)
	os.makedirs(os.path.dirname(save_path2), exist_ok=True)
	Image.fromarray(skeletonArray).save(save_path2)
	os.makedirs(os.path.dirname(save_path3), exist_ok=True)
	Image.fromarray(skeletonArrayInverted).save(save_path3)

begin = time.time()
#chaincode("Image_(1).bmp")
skeleton("Image_(3).bmp")
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")
