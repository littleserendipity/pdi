from PIL import Image
import numpy
import time
import os

def isPixelValid(arr, y, x):
    if ((y >= 0 and y < len(arr)) and (x >= 0 and x < len(arr[0]))):
        return True
    else:
        return False

def getWindowAVG(arr, y, x):
    values = []

    for y2 in range(-1, 2):
        for x2 in range(-1, 2):
            temp_y = y + y2
            temp_x = x + x2

            if (isPixelValid(arr, temp_y, temp_x)):
                values.append(arr[temp_y, temp_x])

    l_avg = [int(numpy.mean(values)), numpy.std(values)]
    return l_avg

def agucar(img):
    path = "agucar"
    save_path = os.path.join(path, img.replace(".", "_result."))

    im = Image.open(os.path.join(path, img), "r")
    width, height = im.size

    arr = numpy.array(im, dtype=numpy.uint8)
    n_arr = numpy.zeros((height, width), dtype=numpy.uint8)

    for y in range(height):
        for x in range(width):
            
            if (n_arr[y, x] < 0):
                n_arr[y, x] = 0
            elif (n_arr[y, x] > 255):
                n_arr[y, x] = 255
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(n_arr).show()
    # Image.fromarray(n_arr).save(save_path)

begin = time.time()
agucar("Agucar_(1).jpg")
# agucar("Agucar_(2).jpg")
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")