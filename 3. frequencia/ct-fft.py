from PIL import Image
import numpy
import math
import os

### Cooley-Tukey Fast Fourier Transform ###
def fft(x):
    n = len(x)
    if (n == 1):
        return x
    f_even, f_odd = fft(x[0::2]), fft(x[1::2])
    s = [0] * n
    for m in range(n//2):
        omega = numpy.exp((-2j * numpy.pi * m) / n)
        s[m] = f_even[m] + omega * f_odd[m]
        s[m + n//2] = f_even[m] - omega * f_odd[m]
    return s

def pad2(x):
   m, n = numpy.shape(x)
   M, N = 2 ** int(math.ceil(math.log(m, 2))), 2 ** int(math.ceil(math.log(n, 2)))
   F = numpy.zeros((M,N), dtype=x.dtype)
   F[0:m, 0:n] = x
   return F, m, n

def fft2(f):
   f, m, n = pad2(f)
   return numpy.transpose(fft(numpy.transpose(fft(f)))), m, n

def ifft2(F, m, n):
    f, M, N = fft2(numpy.conj(F))
    f = numpy.matrix(numpy.real(numpy.conj(f)))/(M*N)
    return f[0:m, 0:n]

def fourier(img):
    arr = openImg(img)

    f, m, n = fft2(arr)
    # saveImg(img, numpy.abs(f), "frequence")

    f = numpy.conj(laplace((numpy.conj(f))))
    # saveImg(img, numpy.abs(f), "laplace")

    i = ifft2(f, m, n)
    saveImg(img, numpy.abs(i), "invert")


### Utils ###
def laplace(arr):
    h, w = numpy.shape(arr)
    n_arr = numpy.zeros((h, w), arr.dtype)

    kernel = numpy.array([
        [0,  1/4, 0], 
        [1/4, -2, 1/4], 
        [0,  1/4, 0],
    ])

    for y in range(h):
        for x in range(w):
            # n_arr[y,x] = (arr[y,x] - sharpering(kernel, arr, y, x))
            n_arr[y,x] = (sharpering(kernel, arr, y, x))
    return n_arr

def sharpering(kernel, arr, y, x):
    total = 0
    begin = -(len(kernel)//2)
    end = len(kernel[0])//2
    for y2 in range(begin, end+1):
        for x2 in range(begin, end+1):
            temp_y = y + y2
            temp_x = x + x2
            if (isIndexValid(arr, temp_y, temp_x)):
                kernel_x = y2 - begin
                kernel_y = x2 - begin
                total += (kernel[kernel_y, kernel_x] * arr[temp_y, temp_x])
    return total

def isIndexValid(arr, y, x):
    if ((y >= 0 and y < len(arr)) and (x >= 0 and x < len(arr[0]))):
        return True
    return False

def pixelRange(arr):
    for y in range(len(arr)):
        for x in range(len(arr[0])):
            arr[y,x] = int(numpy.minimum(0, numpy.maximum(0, arr[y,x])))
    return arr

def openImg(img):
    im = Image.open(img, "r")
    return numpy.array(im, dtype=numpy.uint64)

def saveImg(img, n_arr, extension=""):
    os.makedirs(os.path.dirname(img), exist_ok=True)
    ext = "_result_" + extension + "."
    Image.fromarray(n_arr).convert("L").show()
    # Image.fromarray(n_arr).convert("L").save(img.replace(".", ext))


def main():
    img = [
        "Agucar_(1).jpg",
        # "Agucar_(2).jpg",
        "Agucar_(3).jpg",
        # "Agucar_(4).jpg",
        # "Agucar_(5).jpg",
    ]
    for x in range(len(img)):
        fourier(os.path.join("images", img[x]))

if __name__ == "__main__":
    main()