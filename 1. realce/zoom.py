from PIL import Image
import numpy
import os

def zoom(img, n_width, n_height):
    path = "./zoom/" + img
    save_path = path + "result/" + img

    im = Image.open(path, "r")
    arr = numpy.array(im)

    width = len(arr[0])
    height = len(arr)
    n_arr = numpy.zeros((n_height, n_width) , dtype=numpy.uint8)

    if (n_width > width):
        f_width = int(numpy.ceil((n_width / width)))
    else:
        f_width = 1

    if (n_height > height):
        f_height = int(numpy.ceil((n_height / height)))
    else:
        f_height = 1

    for y in range(len(arr)):
        new_y = (y * f_height)

        for x in range(len(arr[y])):
            new_x = (x * f_width)

            for temp_y in range(f_height):
                for temp_x in range(f_width):
                    if (new_y + temp_y < n_height) and (new_x + temp_x < n_width):
                        n_arr[(new_y + temp_y), (new_x + temp_x)] = arr[y, x]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(n_arr).save(save_path)

# zoom("Zoom_in_(1).jpg", 360, 480)
# zoom("Zoom_in_(2).jpg", 2592, 1456)
# zoom("Zoom_in_(3).jpg", 720, 990)

# zoom("Zoom_out_(1).jpg", 271, 120)
zoom("Zoom_out_(2).jpg", 317, 500)
# zoom("Zoom_out_(3).jpg", 174, 500)
