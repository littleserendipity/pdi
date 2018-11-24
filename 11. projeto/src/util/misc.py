import numpy as np

def random_split_dataset(images, labels, percent):
    v_images = list()
    v_labels = list()
    validation_size = round(percent * len(images), -1)

    t_images = list(images)
    g_labels = list(labels)

    while (len(v_images) < validation_size):
        index = int(np.random.randint(len(t_images), size=1))
        v_images.append(t_images.pop(index))
        v_labels.append(g_labels.pop(index))

    return t_images, g_labels, v_images, v_labels

def epochs_and_steps(g_total, v_total, percent):
    epochs = int(np.ceil(percent*g_total))
    steps_per_epoch = int(g_total/epochs)
    validation_steps = int(v_total/epochs)
    return epochs, steps_per_epoch, validation_steps