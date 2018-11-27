import numpy as np
import math

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

def epochs_and_steps(g_total, v_total=None):
    if (v_total == 0):
        g_divisor = int(g_total * 0.1)
        v_total = 0
    else:
        g_divisor, _ = middle_cdr(g_total, v_total)

    epochs = g_total//g_divisor
    steps_per_epoch = g_total//epochs
    validation_steps = v_total//epochs

    return epochs, steps_per_epoch, validation_steps

def round_up(x, digit=10):
    return x if (x % digit == 0) else (x + digit) - (x % digit)

def round_down(x):
    x = int(x)
    return round(x, 1-len(str(x)))

def middle_cdr(a, b):
    divisors_a = divisors(a)
    divisors_b = divisors(b)
    l = [[[i,j] for i in divisors_a if (a//i == b//j)] for j in divisors_b]
    return l[len(l)//2][0]

def divisors(n):
    divs = [1]
    for i in range(2,int(math.sqrt(n))+1):
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

def clip(arr, min_limit, max_limit):
    arr[arr < min_limit] = min_limit
    arr[arr > max_limit] = max_limit
    return arr