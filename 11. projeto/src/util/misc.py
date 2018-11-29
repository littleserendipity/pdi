import numpy as np

def random_split_dataset(images, labels, percent):
    v_images = list()
    v_labels = list()
    validation_size = round_down(percent*len(images))

    t_images = list(images)
    g_labels = list(labels)

    while (len(v_images) < validation_size):
        index = int(np.random.randint(len(t_images), size=1))
        v_images.append(t_images.pop(index))
        v_labels.append(g_labels.pop(index))

    return t_images, g_labels, v_images, v_labels

def epochs_and_steps(len_data, len_validation=None):
    if (len_validation == 0):
        g_divisor = int(len_data * 0.1)
    else:
        g_divisor = middle_cdr(len_data, len_validation)

    epochs = len_data//g_divisor
    steps_per_epoch = len_data//epochs
    validation_steps = len_validation//epochs

    return epochs, steps_per_epoch, validation_steps

def round_up(x, digit=10):
    return int(x) if (x % digit == 0) else int((x + digit) - (x % digit))

def round_down(x):
    x = int(x)
    return round(x, 1-len(str(x)))

def middle_cdr(a, b):
    divisors_a = divisors(a)
    divisors_b = divisors(b)
    l = [(i, j) for i in divisors_a for j in divisors_b if (a//i == b//j)]
    index = int((len(l)//2) + 1)
    return l[index][0]

def divisors(n):
    divs = [1]
    for i in range(2,int(np.sqrt(n))+1):
        if n%i == 0:
            divs.extend([i,n//i])
    divs.extend([n])
    return sorted(list(set(divs)))

def str_center(*arr):
    stringfy = lambda arr: [str(x) for x in arr]
    max_length = lambda arr: len(max(arr, key=len))
    padding = lambda arr, pad: [x.center(pad) for x in arr]

    arr = stringfy(arr)
    length = max_length(arr)
    arr = padding(arr, length)

    return tuple(arr)