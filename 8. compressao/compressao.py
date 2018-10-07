from PIL import Image
import numpy
import math
import time
import os

def RLE(binArray):
	height, width = binArray.shape
	arrayCount = [[] for i in range(height)]
	count = 0
	valueToCheck = 1 # start as white
	for yy in range(height):
		for xx in range(width):
			if(valueToCheck == binArray[yy,xx]):
				count+=1
			else:
				arrayCount[yy].append(count)
				count = 1
				valueToCheck = binArray[yy, xx]
		arrayCount[yy].append(count)
	#print(arrayCount)
	arrayCount = numpy.array(arrayCount)
	#arrayCount = numpy.array([numpy.array(xi) for xi in arrayCount])   # it will become an array of arrays, but it'll increase in size, sigh
	#print(binArray[0])
	#print(arrayCount[0])
	return arrayCount

def RLETest(binArray):
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
	arrayCount = numpy.array(arrayCount, dtype = numpy.uint8)
	print(arrayCount)
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
	im = Image.open(os.path.join(path, img), "r")
	imageArray = numpy.array(im, dtype=numpy.uint8)
	#print(len(imageArray.shape))
	if(len(imageArray.shape) == 3):
		imageArray = imageArray[:, :, 0]
	#print(imageArray[0])
	width, height = im.size
	g7 = numpy.zeros((height, width), dtype=numpy.uint8)
	g6 = numpy.zeros((height, width), dtype=numpy.uint8)
	g5 = numpy.zeros((height, width), dtype=numpy.uint8)
	g4 = numpy.zeros((height, width), dtype=numpy.uint8)
	g3 = numpy.zeros((height, width), dtype=numpy.uint8)
	g2 = numpy.zeros((height, width), dtype=numpy.uint8)
	g1 = numpy.zeros((height, width), dtype=numpy.uint8)
	g0 = numpy.zeros((height, width), dtype=numpy.uint8)
	print(numpy.array([1, 1, 1]))
	for yy in range(height):
		for xx in range(width):
			valueBin = bin(imageArray[yy, xx])[2:].zfill(8)
			#print(valueBin)
			g7[yy, xx] = int(valueBin[7])
			g6[yy, xx] = int(valueBin[6])
			g5[yy, xx] = int(valueBin[5])
			g4[yy, xx] = int(valueBin[4])
			g3[yy, xx] = int(valueBin[3])
			g2[yy, xx] = int(valueBin[2])
			g1[yy, xx] = int(valueBin[1])
			g0[yy, xx] = int(valueBin[0])	
	# g0RLE = RLE(g0)
	# g1RLE = RLE(g1)
	# g2RLE = RLE(g2)
	# g3RLE = RLE(g3)
	# g4RLE = RLE(g4)
	# g5RLE = RLE(g5)
	# g6RLE = RLE(g6)
	# g7RLE = RLE(g7)
	g0RLE = RLETest(g0)
	g1RLE = RLETest(g1)
	g2RLE = RLETest(g2)
	g3RLE = RLETest(g3)
	g4RLE = RLETest(g4)
	g5RLE = RLETest(g5)
	g6RLE = RLETest(g6)
	g7RLE = RLETest(g7)
	numpy.save("image.npy", imageArray)
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
	numpy.save("g7RLE.npy", g7RLE)
	# numpy.save("g1.npy", g1)
	# numpy.save("g1RLE.npy", g1RLE)
	# numpy.save("g2.npy", g2)
	# numpy.save("g2RLE.npy", g2RLE)
	# numpy.save("g3.npy", g3)
	# numpy.save("g3RLE.npy", g3RLE)
	# numpy.save("g4.npy", g4)
	# numpy.save("g4RLE.npy", g4RLE)
	# numpy.save("g5.npy", g5)
	# numpy.save("g5RLE.npy", g5RLE)
	# numpy.save("g6.npy", g6)
	# numpy.save("g6RLE.npy", g6RLE)
	# numpy.save("g7.npy", g7)
	# numpy.save("g7RLE.npy", g7RLE)
	# g0[g0 > 0] = 255
	# g1[g1 > 0] = 255
	# g2[g2 > 0] = 255
	# g3[g3 > 0] = 255
	# g4[g4 > 0] = 255
	# g5[g5 > 0] = 255
	# g6[g6 > 0] = 255
	# g7[g7 > 0] = 255
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


begin = time.time()
binaryBitplan("Image_(1).tif")
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")
