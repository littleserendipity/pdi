from PIL import Image
from numpy import array

path = "./realce/Clarear_(1).jpg"

im = Image.open(path)
rgb_im = im.convert('RGB')

arr = array(rgb_im)

for y in range(len(arr)):
    for x in range(len(arr[y])):
        if (y % 2 == 0) and (x % 2 == 1):
            arr[y,x,1] = 255

Image.fromarray(arr).show()