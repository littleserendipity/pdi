from PIL import Image
import numpy
import time
import os

def isIndexValid(arr, y, x):
    if ((y >= 0 and y < len(arr)) and (x >= 0 and x < len(arr[0]))):
        return True
    else:
        return False

def getPixelRange(pixel):
    return numpy.minimum(255, numpy.maximum(0, pixel))

def sharpering(arr, y, x):
    total = 0
    mask = numpy.array([
        [-1, -1, -1], 
        [-1, 9, -1], 
        [-1, -1, -1]
    ])

    begin = (0 - int(len(mask)/2))
    end = (1 + int(len(mask[0])/2))

    for y2 in range(begin, end):
        for x2 in range(begin, end):
            temp_y = y + y2
            temp_x = x + x2

            if (isIndexValid(arr, temp_y, temp_x)):
                mask_x = y2 + len(mask) - begin - end - 1
                mask_y = x2 + len(mask) - begin - end - 1
                total += (mask[mask_y, mask_x] * arr[temp_y, temp_x])

    return getPixelRange( total )

def agucar(img):
    path = "agucar"
    save_path = os.path.join(path, img.replace(".", "_result."))

    im = Image.open(os.path.join(path, img), "r")
    width, height = im.size

    arr = numpy.array(im, dtype=numpy.uint8)
    n_arr = numpy.zeros((height, width), dtype=numpy.uint8)

    for y in range(height):
        for x in range(width):
            n_arr[y, x] = arr[y,x] + sharpering(arr, y, x)
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(n_arr).show()
    # Image.fromarray(n_arr).save(save_path)

begin = time.time()
agucar("Agucar_(1).jpg")
# agucar("Agucar_(2).jpg")
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")