import keras
import numpy as np

from keras import layers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

from sklearn import preprocessing
from sklearn.preprocessing import normalize


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train = X_train.reshape([-1, 28, 28, 1]).astype('float32')
X_test = X_test.reshape([-1, 28, 28, 1]).astype('float32')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def le_net5(*args, **kwargs):
    """Returns le-net5 model object (created by keras)"""
    
    model = keras.Sequential()
    
    model.add(layers.Conv2D(6, (5, 5), strides=1, input_shape=(28, 28, 1),  activation='relu'))
    model.add(layers.MaxPool2D((2, 2), strides=2))
    
    model.add(layers.Conv2D(16, (5, 5), strides=1,  activation='relu'))
    model.add(layers.MaxPool2D((2, 2), strides=2))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation = 'softmax'))

    return model


model = le_net5()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
)
