import numpy as np
from keras.datasets import mnist
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
import matplotlib.pyplot as plt


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'Training Data: {x_train.shape}')
print(f'Training Labels: {y_train.shape}')
print(f'Labels: {y_train}')

width, height, channels = x_train.shape[1], x_train.shape[2], 1
x_train = x_train.reshape((x_train.shape[0], width, height, channels))
x_test = x_test.reshape((x_test.shape[0], width, height, channels))

# One hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# confirm scale of pixels
print(f'Train min={x_train.min()} max={x_train.max()}')
print(f'Train min={x_test.min()} max={x_test.max()}')

# prepare an iterators to scale images
train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)


idx = 1
img = x_train[idx]
label = y_train[idx]
print(img.shape)
plt.imshow(img)
print("label", label)