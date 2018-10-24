#from PIL import Image
import numpy as np
import math
import time
import os
import h5py
from keras.models import Sequential 
from keras.layers import Convolution2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

def load_dataset():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T     # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T        # Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.





# classifier = Sequential()
# classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Flatten())
# classifier.add(Dense(209, activation='sigmoid'))
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




# print("Train X Orig Shape: " + str(train_x_orig.shape))
# print("Train Y Shape: " + str(train_y[0].shape))
# classifier.fit(train_x_orig, train_y[0])

#print("Array x test shape: " + str(test_x_orig.shape))
#array_x_test = test_x_orig[0].reshape(-1, 64, 64, 3)

# test_predictions = classifier.predict(test_x_orig)
# test_predictions = np.round(test_predictions)
# print(test_y.shape)
# print(test_predictions.shape)
# accuracy = accuracy_score(test_y, test_predictions)
# print("Accuracy: " + str(accuracy))

# from keras.preprocessing.image import ImageDataGenerator

# classifier.fit_generator(
#     train_x_orig,
#     steps_per_epoch=8000,
#     epochs=10,
#     validation_data=test_x_orig,
#     validation_steps=800
# )

model = Sequential()
#model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
numLayers = 4
minNeurons = 20
maxNeurons = 209
kernel = (3, 3)
steps = np.floor(maxNeurons / (numLayers + 1))
neurons = np.arange(minNeurons, maxNeurons, steps)
neurons = neurons.astype(np.int32)
EPOCHS = 20
BATCH_SIZE = 50
# print(neurons)


#print(train_x_orig[0])
# print(trainXPlus[0])



for i in range(numLayers):
    if (i == 0):
        shape = (64, 64, 3)
        model.add(Convolution2D(neurons[i], kernel, input_shape=shape))
    else:
        model.add(Convolution2D(neurons[i], kernel))
    model.add(Activation("relu"))
    #model.add(LeakyReLU(alpha=0.3))

#model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(209))
#model.add(Activation("relu"))
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# model.summary()
# print(train_x_orig.shape)
# print(train_x_orig)
#print(trainXScaled)
trainXScaled = train_x_orig/255
model.fit(trainXScaled, train_y[0], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)


# train_x_flatten2 = train_x_orig.reshape(-1, train_x_orig.shape[0]).T 
# print(train_x_flatten2.shape)
# model.fit(train_x_flatten2, train_y[0], epochs=10, batch_size=20)

testXScaled = test_x_orig/255
test_predictions = model.predict(testXScaled, batch_size=BATCH_SIZE)
test_predictions = np.round(test_predictions)

# test_x_flatten2 = test_x_orig.reshape(-1, test_x_orig.shape[0]).T 

# test_predictions = model.predict(test_x_flatten2)
# #test_predictions = np.round(test_predictions)

# print(test_y[0])
# print(test_predictions)
accuracy = accuracy_score(test_y[0], test_predictions)

print("Accuracy: " + str(accuracy))

test_predictions2 = model.predict(trainXScaled, batch_size=BATCH_SIZE)
test_predictions2 = np.round(test_predictions2)

accuracy2 = accuracy_score(train_y[0], test_predictions2)
print("Accuracy same images: " + str(accuracy2))


# Below: Trying to expand the training set, but the accuracy result was reaaaally low(34%) and idk why, so I've given up on it

# datageneratorTrain = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# datageneratorTest = ImageDataGenerator(rescale=1./255)

# trainXPlus = datageneratorTrain.flow(train_x_orig, train_y[0])
# testXPlus = datageneratorTest.flow(test_x_orig, test_y[0])


# model.fit_generator(
#     trainXPlus,
#     steps_per_epoch=20,
#     epochs=EPOCHS
# )

# testXScaled = test_x_orig/255
# test_predictions = model.predict(testXScaled, batch_size=BATCH_SIZE)
# test_predictions = np.round(test_predictions)
# accuracy = accuracy_score(test_y[0], test_predictions)
# print("Accuracy: " + str(accuracy))

# This test bellow doesn't work because the array sizes are different

# testPredictionsGenerator = model.predict_generator(
#     testXPlus,
#     steps=20
# )
# accuracyGenerator = accuracy_score(test_y[0], testPredictionsGenerator)
# print("Accuracy Generator: " + str(accuracyGenerator))

