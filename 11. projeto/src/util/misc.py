import numpy as np

def random_split_dataset(images, labels, split):
    v_images = list()
    v_labels = list()
    validation_size = split * len(images)

    g_images = list(images)
    g_labels = list(labels)

    while (len(v_images) < validation_size):
        index = int(np.random.randint(len(g_images), size=1))
        v_images.append(g_images.pop(index))
        v_labels.append(g_labels.pop(index))

    return g_images, g_labels, v_images, v_labels

def epochs_and_steps(total, percent):
    epochs = int(np.ceil(percent*total))
    steps_per_epoch = int(total/epochs)
    return epochs, steps_per_epoch