from PIL import Image
import numpy
import math
import time
import os


def RLEDecode(rleArray):
	#print(rleArray)
	size = rleArray.size
	#print(size)
	arrayLine = []
	arrayTotal = []
	value = True # start as white
	x = 0
	#print(rleArray[0])
	for i in range(size):
		if(rleArray[i] < 255): # 255 = new line
			for v in range(rleArray[i]):
				arrayLine.append(value)
			value = not value
		else: #if it's a new line
			arrayTotal.append((arrayLine))
			arrayLine = []
			value = not value
	arrayTotal.append((arrayLine)) # last line doesn't have a 255 on the end
	# print(len(arrayTotal[0]))
	# print(len(arrayTotal))
	width = len(arrayTotal[0])
	height = len(arrayTotal)
	newArray = numpy.zeros((height, width), dtype=numpy.uint8)
	for yy in range(height):
		for xx in range(width):
			newArray[yy, xx] = int(arrayTotal[yy][xx])

	# print(newArray)
	# print(rleArray)
	return newArray


def RLE(binArray):
	height, width = binArray.shape
	arrayCount = []
	count = 0
	valueToCheck = 1 # start as white
	#255 = linebreak
	#goes only from values 0 to 254
	for yy in range(height):
		for xx in range(width):
			if(valueToCheck == binArray[yy,xx]):
				if(count == 254): #max value
					arrayCount.append(count)
					count = 0
					arrayCount.append(count)
				count+=1
			else:
				arrayCount.append(count)
				count = 1
				valueToCheck = binArray[yy, xx]
		arrayCount.append(count)
		if(yy < height-1):
			arrayCount.append(255) #new line
			count = 0
	arrayCount = numpy.array(arrayCount, dtype = numpy.uint8)
	#print(arrayCount)
	return arrayCount

def binaryBitplan(img):
	path = "Images"
	# save_path7 = os.path.join(path, "result_binaryBitplan", "g7_" + img[:-4] + ".png")
	# save_path6 = os.path.join(path, "result_binaryBitplan", "g6_" + img[:-4] + ".png")
	# save_path5 = os.path.join(path, "result_binaryBitplan", "g5_" + img[:-4] + ".png")
	# save_path4 = os.path.join(path, "result_binaryBitplan", "g4_" + img[:-4] + ".png")
	# save_path3 = os.path.join(path, "result_binaryBitplan", "g3_" + img[:-4] + ".png")
	# save_path2 = os.path.join(path, "result_binaryBitplan", "g2_" + img[:-4] + ".png")
	# save_path1 = os.path.join(path, "result_binaryBitplan", "g1_" + img[:-4] + ".png")
	# save_path0 = os.path.join(path, "result_binaryBitplan", "g0_" + img[:-4] + ".png")
	# save_path = os.path.join(path, "result_binaryBitplan", "resultA7toA2_" + img[:-4] + ".png")
	# save_path2 = os.path.join(path, "result_binaryBitplan", "resultG7toG2_" + img[:-4] + ".png")
	im = Image.open(os.path.join(path, img), "r")
	imageArray = numpy.array(im, dtype=numpy.uint8)
	#print(len(imageArray.shape))
	if(len(imageArray.shape) == 3):
		imageArray = imageArray[:, :, 0]
	#print(imageArray[0])
	width, height = im.size
	#newImageArray = numpy.zeros((height, width), dtype=numpy.uint8)
	a7 = numpy.zeros((height, width), dtype=numpy.bool)
	a6 = numpy.zeros((height, width), dtype=numpy.bool)
	a5 = numpy.zeros((height, width), dtype=numpy.bool)
	a4 = numpy.zeros((height, width), dtype=numpy.bool)
	a3 = numpy.zeros((height, width), dtype=numpy.bool)
	a2 = numpy.zeros((height, width), dtype=numpy.bool)
	a1 = numpy.zeros((height, width), dtype=numpy.bool)
	a0 = numpy.zeros((height, width), dtype=numpy.bool)
	g7 = numpy.zeros((height, width), dtype=numpy.bool)
	g6 = numpy.zeros((height, width), dtype=numpy.bool)
	g5 = numpy.zeros((height, width), dtype=numpy.bool)
	g4 = numpy.zeros((height, width), dtype=numpy.bool)
	g3 = numpy.zeros((height, width), dtype=numpy.bool)
	g2 = numpy.zeros((height, width), dtype=numpy.bool)
	g1 = numpy.zeros((height, width), dtype=numpy.bool)
	g0 = numpy.zeros((height, width), dtype=numpy.bool)
	for yy in range(height):
		for xx in range(width):
			valueBin = bin(imageArray[yy, xx])[2:].zfill(8)
			#print(valueBin)
			a0[yy, xx] = bool(int(valueBin[7]))
			a1[yy, xx] = bool(int(valueBin[6]))
			a2[yy, xx] = bool(int(valueBin[5]))
			a3[yy, xx] = bool(int(valueBin[4]))
			a4[yy, xx] = bool(int(valueBin[3]))
			a5[yy, xx] = bool(int(valueBin[2]))
			a6[yy, xx] = bool(int(valueBin[1]))
			a7[yy, xx] = bool(int(valueBin[0]))	

			g7[yy, xx] = a7[yy, xx]
			g6[yy, xx] = XOR(a6[yy, xx], a7[yy, xx])
			g5[yy, xx] = XOR(a5[yy, xx], a6[yy, xx])
			g4[yy, xx] = XOR(a4[yy, xx], a5[yy, xx])
			g3[yy, xx] = XOR(a3[yy, xx], a4[yy, xx])
			g2[yy, xx] = XOR(a2[yy, xx], a3[yy, xx])
			g1[yy, xx] = XOR(a1[yy, xx], a2[yy, xx])
			g0[yy, xx] = XOR(a0[yy, xx], a1[yy, xx])
	# g0RLE = RLE(g0)
	# g1RLE = RLE(g1)
	# g2RLE = RLE(g2)
	# g3RLE = RLE(g3)
	# g4RLE = RLE(g4)
	# g5RLE = RLE(g5)
	# g6RLE = RLE(g6)
	# g7RLE = RLE(g7)

	a0RLE = RLE(a0)
	a1RLE = RLE(a1)
	a2RLE = RLE(a2)
	a3RLE = RLE(a3)
	a4RLE = RLE(a4)
	a5RLE = RLE(a5)
	a6RLE = RLE(a6)
	a7RLE = RLE(a7)

	g0RLE = RLE(g0)
	g1RLE = RLE(g1)
	g2RLE = RLE(g2)
	g3RLE = RLE(g3)
	g4RLE = RLE(g4)
	g5RLE = RLE(g5)
	g6RLE = RLE(g6)
	g7RLE = RLE(g7)
	

	# print(g7)
	# print(g7RLE)
	# g0 = numpy.packbits(g0, axis=None)
	# g1 = numpy.packbits(g1, axis=None)
	# g2 = numpy.packbits(g2, axis=None)
	# g3 = numpy.packbits(g3, axis=None)
	# g4 = numpy.packbits(g4, axis=None)
	# g5 = numpy.packbits(g5, axis=None)
	# g6 = numpy.packbits(g6, axis=None)
	# g7 = numpy.packbits(g7, axis=None)
	#print(len(g0.tostring()))

	# numpy.save("image.npy", imageArray)
	numpy.save("g0.npy", g0)
	numpy.save("g0RLE.npy", g0RLE)
	numpy.save("g1.npy", g1)
	numpy.save("g1RLE.npy", g1RLE)
	numpy.save("g2.npy", g2)
	numpy.save("g2RLE.npy", g2RLE)
	numpy.save("g3.npy", g3)
	numpy.save("g3RLE.npy", g3RLE)
	numpy.save("g4.npy", g4)
	numpy.save("g4RLE.npy", g4RLE)
	numpy.save("g5.npy", g5)
	numpy.save("g5RLE.npy", g5RLE)
	numpy.save("g6.npy", g6)
	numpy.save("g6RLE.npy", g6RLE)
	numpy.save("g7.npy", g7)
	#numpy.savetxt("g7.txt", g7)
	numpy.save("g7RLE.npy", g7RLE)
	#numpy.savetxt("g7RLE.txt", g7RLE)

	numpy.save("a0.npy", a0)
	numpy.save("a0RLE.npy", a0RLE)
	numpy.save("a1.npy", a1)
	numpy.save("a1RLE.npy", a1RLE)
	numpy.save("a2.npy", a2)
	numpy.save("a2RLE.npy", a2RLE)
	numpy.save("a3.npy", a3)
	numpy.save("a3RLE.npy", a3RLE)
	numpy.save("a4.npy", a4)
	numpy.save("a4RLE.npy", a4RLE)
	numpy.save("a5.npy", a5)
	numpy.save("a5RLE.npy", a5RLE)
	numpy.save("a6.npy", a6)
	numpy.save("a6RLE.npy", a6RLE)
	numpy.save("a7.npy", a7)
	numpy.save("a7RLE.npy", a7RLE)
	#numpy.savetxt("a7.txt", a7, fmt="%d", newline=",")
	#numpy.savetxt("a7RLE.txt", a7RLE, fmt="%d", newline=",")
	# g0[g0 > 0] = 255
	# g1[g1 > 0] = 255
	# g2[g2 > 0] = 255
	# g3[g3 > 0] = 255
	# g4[g4 > 0] = 255
	# g5[g5 > 0] = 255
	# g6[g6 > 0] = 255
	# g7[g7 > 0] = 255
	# for yy in range(height):
	# 	for xx in range(width):
	# 		newImageArray[yy, xx] = (a7[yy, xx] * 128) + (a6[yy, xx] * 64) + (a5[yy, xx] * 32) + (a4[yy,xx] * 16) + (a3[yy, xx] * 8) + (a2[yy, xx] * 4)
	# os.makedirs(os.path.dirname(save_path), exist_ok=True)
	# Image.fromarray(newImageArray).save(save_path)

	# newImageArray2 = decodeGrayCode(height, width, g7, g6, g5, g4, g3, g2, g1, g0)

	# os.makedirs(os.path.dirname(save_path2), exist_ok=True)
	# Image.fromarray(newImageArray2).save(save_path2)
	# os.makedirs(os.path.dirname(save_path0), exist_ok=True)
	# Image.fromarray(g0).save(save_path0)

	# os.makedirs(os.path.dirname(save_path1), exist_ok=True)
	# Image.fromarray(g1).save(save_path1)

	# os.makedirs(os.path.dirname(save_path2), exist_ok=True)
	# Image.fromarray(g2).save(save_path2)

	# os.makedirs(os.path.dirname(save_path3), exist_ok=True)
	# Image.fromarray(g3).save(save_path3)

	# os.makedirs(os.path.dirname(save_path4), exist_ok=True)
	# Image.fromarray(g4).save(save_path4)

	# os.makedirs(os.path.dirname(save_path5), exist_ok=True)
	# Image.fromarray(g5).save(save_path5)

	# os.makedirs(os.path.dirname(save_path6), exist_ok=True)
	# Image.fromarray(g6).save(save_path6)

	# os.makedirs(os.path.dirname(save_path7), exist_ok=True)
	# Image.fromarray(g7).save(save_path7)

def decodeImageBinary(name, decode):
	path = "Images"
	if(decode):
		a7 = RLEDecode(numpy.load("a7RLE.npy"))
		print(a7[1])
		a7Normal = numpy.load("a7.npy")
		print(a7Normal[1])
		a6 = RLEDecode(numpy.load("a6RLE.npy"))
		a5 = RLEDecode(numpy.load("a5RLE.npy"))
		a4 = RLEDecode(numpy.load("a4RLE.npy"))
		a3 = RLEDecode(numpy.load("a3RLE.npy"))
		a2 = RLEDecode(numpy.load("a2RLE.npy"))
		a1 = RLEDecode(numpy.load("a1RLE.npy"))
		a0 = RLEDecode(numpy.load("a0RLE.npy"))
	else:
		a7 = numpy.load("a7.npy")
		a6 = numpy.load("a6.npy")
		a5 = numpy.load("a5.npy")
		a4 = numpy.load("a4.npy")
		a3 = numpy.load("a3.npy")
		a2 = numpy.load("a2.npy")
		a1 = numpy.load("a1.npy")
		a0 = numpy.load("a0.npy")
	height, width = a7.shape
	newImageArray = numpy.zeros((height, width), dtype=numpy.uint8)
	save_path = os.path.join(path, name + ".png")
	for yy in range(height):
		for xx in range(width):
			newImageArray[yy, xx] = (a7[yy, xx] * 128) + (a6[yy, xx] * 64) + (a5[yy, xx] * 32) + (a4[yy,xx] * 16) + (a3[yy, xx] * 8) + (a2[yy, xx] * 4) + (a1[yy, xx] * 2) + (a0[yy, xx])
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	Image.fromarray(newImageArray).save(save_path)


#def decodeImageGray(name):

def XOR(arg1, arg2):
	ret = not(arg1 and arg2) and (arg1 or arg2)
	return ret


def decodeGrayCode(height, width, g7, g6, g5, g4, g3, g2, g1, g0):
	newImageArray2 = numpy.zeros((height, width), dtype=numpy.uint8)
	for yy in range(height):
		for xx in range(width):
			a7 = g7[yy, xx]
			a6 = XOR(g6[yy, xx], a7)
			a5 = XOR(g5[yy, xx], a6)
			a4 = XOR(g4[yy, xx], a5)
			a3 = XOR(g3[yy, xx], a4)
			a2 = XOR(g2[yy, xx], a3)
			a1 = XOR(g1[yy, xx], a2)
			a0 = XOR(g0[yy, xx], a1)
			newImageArray2[yy, xx] = (a7 * 128) + (a6 * 64) + (a5 * 32) + (a4 * 16) + (a3 * 8) + (a2 * 4) + (a1 * 2) + a0
	return newImageArray2


begin = time.time()
#binaryBitplan("Image_(1).tif")
#decodeImageBinary("testDecode", True)
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")
