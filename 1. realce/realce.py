from PIL import Image
import numpy
import time
import os

def getMinMax(arr):
    return [min([x for x in arr if x !=0]), max(arr)]

def histogram(arr):
    h_arr = numpy.zeros((256), dtype=numpy.uint8)
    g_arr = ["" for x in range(256)]

    for x in range(len(arr)):
        h_arr[arr[x]] += 1

    for x in range(len(h_arr)):
        g_arr[x] = (str(x) + "\t: " + str("|" * int(h_arr[x] * 0.25)) + "\n")
    
    return [g_arr, getMinMax(h_arr)]

def printHistogram(save_path, g_arr, p_min_max):
    with open((save_path + ".txt"), "w") as text_file:
        text_file.write("Histograma da imagem: " + save_path + "\n")
        text_file.write("Min e Max (min > 0): " + str(p_min_max) + "\n\n")

        for x in range(len(g_arr)):
            text_file.write(g_arr[x])

def no_linear(min, max, x, exp):
    a = 255/(max - min)
    return int(a*(x **(exp*2)))

def realce(img, exp):
    path = "realce"
    save_path = os.path.join(path, img)

    im = Image.open(os.path.join(path, img), "r")
    l_pixel = list(im.getdata())

    g_arr, p_min_max = histogram(l_pixel)
    printHistogram(save_path, g_arr, getMinMax(l_pixel))

    save_path = os.path.join(path, "result", img)

    for x in range(len(l_pixel)):
        pixel = no_linear(p_min_max[0], p_min_max[1], l_pixel[x], exp)

        if (pixel < p_min_max[0]):
            l_pixel[x] = p_min_max[0]
        elif (pixel > p_min_max[1]):
            l_pixel[x] = p_min_max[1]
        else:
            l_pixel[x] = pixel

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    g_arr, p_min_max = histogram(l_pixel)
    printHistogram(save_path, g_arr, getMinMax(l_pixel))

    im2 = Image.new(im.mode, im.size)
    im2.putdata(l_pixel)
    # im2.show()
    im2.save(save_path)

begin = time.time()
realce("Clarear_(1).jpg", 0.75)
realce("Clarear_(2).jpg", 0.75)
realce("Clarear_(3).jpg", 0.75)

realce("Escurecer_(1).jpg", 0.4)
realce("Escurecer_(2).jpg", 0.4)
realce("Escurecer_(3).jpg", 0.4)
end = time.time()

print("Finalizado: " + str(round(end-begin, 2)) + "s\n")