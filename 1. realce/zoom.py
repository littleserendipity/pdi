from PIL import Image
import numpy

load_path = "./zoom/Zoom_in_(1).jpg"
save_path = "./"

im = Image.open(load_path)
arr = numpy.array(im)

height = len(arr)
width = len(arr[0])

n_height = 480
n_width = 360

f_height = int(n_height / height) 
f_width = int(n_width / width)

dec_height = (n_height / height) - f_height
dec_width = (n_width / width) - f_width

print(f_height)
print(dec_height)
print()
print(f_width)
print(dec_width)

new_arr = numpy.zeros((n_height, n_width))

# rgb_im = im.convert('RGB')
# arr = array(rgb_im)

for y in range(len(arr)):
    new_y = int(y * f_height)

    for x in range(len(arr[y])):
        new_x = int(x * f_width)

        for temp_y in range(f_height):
            for temp_x in range(f_width):
                new_arr[(new_y+temp_y), (new_x+temp_x)] = arr[y, x]
    
Image.fromarray(new_arr).show()