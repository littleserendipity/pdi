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

    f = laplace(f)
    # saveImg(img, numpy.abs(f), "laplace")

    i = ifft2(f, m, n)
    saveImg(img, numpy.abs(i), "invert")


### Utils ###
def laplace(arr):
    h, w = numpy.shape(arr)
    n_arr = numpy.zeros((h, w), arr.dtype)

    # kernel = numpy.array([
    #     1,  1,  1,
    #     1, -8,  1,
    #     1,  1,  1,
    # ])

    kernel = numpy.array([
        1/4,  1/4,  1/4,
        1/4, -8/4,  1/4,
        1/4,  1/4,  1/4,
    ])

    radius = int(numpy.sqrt(len(kernel))/2)

    for y in range(h):
        for x in range(w):
            # n_arr[y,x] = (arr[y,x] - numpy.sum(kernel * getNeighbors(arr, y, x)))
            n_arr[y,x] = numpy.sum(kernel * getNeighbors(arr, y, x, radius))
    return n_arr

def getNeighbors(arr, y, x, radius=1):
    begin, end = -(radius), (radius+1)
    return [getPixel(arr, (y+v), (x+u)) for v in range(begin, end) for u in range(begin, end)]

def getPixel(arr, y, x):
    if ((y >= 0 and y < len(arr)) and (x >= 0 and x < len(arr[0]))):
        return arr[y,x]
    return 0

def openImg(img):
    return numpy.array(Image.open(img, "r"), dtype=numpy.float64)

def saveImg(img, n_arr, extension=""):
    os.makedirs(os.path.dirname(img), exist_ok=True)
    Image.fromarray(n_arr).convert("L").show()
    # Image.fromarray(n_arr).convert("L").save(img.replace(".", ("_result_" + extension + ".")))


### Main ###
def main():
    img = [
        "Agucar_(1).jpg",
        # "Agucar_(2).jpg",
        # "Agucar_(3).jpg",
        # "Agucar_(4).jpg",
        # "Agucar_(5).jpg",
    ]
    for x in range(len(img)):
        fourier(os.path.join("images", img[x]))

if __name__ == "__main__":
    main()