from PIL import Image
import numpy
import time
import os

def getMinMax(arr):
    return [min(arr[arr != 0]), max(arr)]

def histograma(arr):
    h_arr = numpy.zeros((256), dtype=numpy.uint8)

    for x in range(len(arr)):
        h_arr[arr[x]] += 1
        
    for x in range(len(h_arr)):
        print(str(x) + "\t: " + str("|" * int(h_arr[x] * 0.25)))
    
    print("\nMin e Max (min > 0): " + str(getMinMax(h_arr)) + "\n")
    return h_arr

def linear(min, max, x):
    a = 255/(max - min)
    b = (-1 * a * min)
    return (a*x) + b

def no_linear(min, max, x):
    a = 255/(max - min)
    return int(a * (x ** (1/2)))

def realce(option, img):
    path = "realce"
    save_path = os.path.join(path, "result")

    im = Image.open(os.path.join(path, img), "r")
    l_pixel = list(im.getdata())

    h_arr = histograma(l_pixel)
    p_min_max = getMinMax(h_arr)

    for x in range(len(l_pixel)):
        if (option == "c"):
            pixel = linear(p_min_max[0], p_min_max[1], l_pixel[x])
        elif (option == "e"):
            pixel = no_linear(p_min_max[0], p_min_max[1], l_pixel[x])

        if (pixel < p_min_max[0]):
            l_pixel[x] = p_min_max[0]
        elif (pixel > p_min_max[1]):
            l_pixel[x] = p_min_max[1]
        else:
            l_pixel[x] = pixel

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    im2 = Image.new(im.mode, im.size)
    im2.putdata(l_pixel)
    im2.show()
    # im2.save(save_path)

begin = time.time()

# realce("c", "Clarear_(1).jpg")
realce("c", "Clarear_(2).jpg")
# realce("c", "Clarear_(3).jpg")

# realce("e", "Escurecer_(1).jpg")
# realce("e", "Escurecer_(2).jpg")
# realce("e", "Escurecer_(3).jpg")

end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")