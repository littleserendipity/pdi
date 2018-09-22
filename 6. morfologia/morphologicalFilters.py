from PIL import Image
import numpy
import math
import time
import os


def dilation(imageArray, dil=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
	#path = "Images"
	#im = Image.open(os.path.join(path, img), "r")
	#imageArray = numpy.array(im, dtype=numpy.uint8)
	width, height = imageArray.shape
	dil = numpy.array(dil, dtype=numpy.uint8)
	#dil = numpy.ones((5,3), dtype=numpy.uint8)
	print(str(dil))
	widthDIL, heightDIL = dil.shape
	newImageArray = numpy.full((height, width), 255, dtype=numpy.uint8)
	#save_path = os.path.join(path, "result_morph", "resultadoDilatacao_dil=" + str(dil).replace('\n','') + "_" + img)
	for x in range(width):
		for y in range(height):
			if(imageArray[y, x] < 128):
				for xx in range(widthDIL):
					for yy in range(heightDIL):
						if( (dil[xx, yy] > 0) and (yy + y < height) and (xx + x < width)):
							newImageArray[yy+y, xx+x] = 0
	#os.makedirs(os.path.dirname(save_path), exist_ok=True)
	#Image.fromarray(newImageArray).save(save_path)
	return newImageArray

def erosion(imageArray, ero=[[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
	#path = "Images"
	#im = Image.open(os.path.join(path, img), "r")
	#imageArray = numpy.array(im, dtype=numpy.uint8)
	width, height = imageArray.shape
	ero = numpy.array(ero, dtype=numpy.uint8)
	#ero = numpy.ones((5,3), dtype=numpy.uint8)
	print(str(ero))
	widthERO, heightERO = ero.shape
	newImageArray = numpy.full((height, width), 255, dtype=numpy.uint8)
	#save_path = os.path.join(path, "result_morph", "resultadoErosao_dil=" + str(ero).replace('\n','') + "_" + img)
	for x in range(width):
		for y in range(height):
			if(imageArray[y, x] < 128): # should be == 0, but since the image isn't really binary, I have to do this
				willAppear = True
				for xx in range(widthERO):
					for yy in range(heightERO):
						if(ero[xx, yy] > 0):
							if(y + yy < height and x + xx < width):
								if(imageArray[yy+y, xx+x] >= 128): # should be > 0, but since the image isn't really binary, I have to do this   
									willAppear = False
							else:
								willAppear = False
				if(willAppear):
					newImageArray[y, x] = 0
	#os.makedirs(os.path.dirname(save_path), exist_ok=True)
	#Image.fromarray(newImageArray).save(save_path)
	return newImageArray


def removeDots(img, bArray = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
	path = "Images"
	im = Image.open(os.path.join(path, img), "r")
	imageArray = numpy.array(im, dtype=numpy.uint8)
	save_path = os.path.join(path, "result_morph", "resultadoFinalMao2Dilatacoes_array=" + str(bArray).replace('\n','') + ".png")	
	imageErosion = erosion(imageArray, ero=bArray)
	imageFinal = dilation(imageErosion, dil=bArray)
	imageFinal2 = dilation(imageFinal, dil=bArray)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	Image.fromarray(imageFinal2).save(save_path)

def contorno(img, bArray = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]):
	bArray = numpy.ones((3,5), dtype=numpy.uint8)
	path = "Images"
	im = Image.open(os.path.join(path, img), "r")
	imageArray = numpy.array(im, dtype=numpy.uint8)
	save_path = os.path.join(path, "result_morph", "resultadoContornoMao_array=" + str(bArray).replace('\n','') + ".png")
	width, height = imageArray.shape
	imageArrayBinary = numpy.full((height, width), 255, dtype=numpy.uint8)
	for x in range(width):
		for y in range(height):
			if(imageArray[y, x] < 128):
				imageArrayBinary[y, x] = 0
			else:
				imageArrayBinary[y, x] = 255
	print(imageArrayBinary)
	imageErosion = erosion(imageArrayBinary, bArray)
	print(imageErosion)
	imageFinal = numpy.full((height, width), 255, dtype=numpy.uint8)
	for x in range(width):
		for y in range(height):
			imageValue = int(imageArrayBinary[y, x]) - int(imageErosion[y, x])
			print(imageValue)
			if(imageValue < 0):
				imageFinal[y, x] = 0
			else:
				imageFinal[y, x] = 255
	print(imageFinal)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	Image.fromarray(imageFinal).save(save_path)


def blackAndWhite(img):
	path = "Images"
	im = Image.open(os.path.join(path, img), "r")
	imageArray = numpy.array(im, dtype=numpy.uint8)
	save_path = os.path.join(path, "result_morph", "resultadoB&W.png")
	width, height = imageArray.shape
	newImageArray = numpy.zeros((width, height), dtype=numpy.uint8)
	for x in range(width):
		for y in range(height):
			if(imageArray[y, x] < 128):
				newImageArray[y, x] = 0
			else:
				newImageArray[y, x] = 255
	print(newImageArray)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	Image.fromarray(newImageArray).save(save_path)

begin = time.time()
removeDots("Image_(2a).jpg")
#contorno("maoDilatada.png")
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")
