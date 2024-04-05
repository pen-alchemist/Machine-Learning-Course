import keras
import numpy as np

from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

from sklearn import preprocessing
from sklearn.preprocessing import normalize

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


def dataset_classify(features, labels):
    """Returns classified X and y"""
    X0, y0, X1, y1 = [], [], [], []

    for label_ind in range(len(labels)):
        label = labels[label_ind]

        if label < 5:
            X0.append(features[label_ind])
            y0.append(label)
        else:
            X1.append(features[label_ind])
            y1.append(label)

    X0 = np.array(X0)
    y0 = np.array(y0)
    X1 = np.array(X1)
    y1 = np.array(y1)

    return X0, y0, X1, y1


def le_net5(*args, **kwargs):
    """Returns le-net5 model object (created by keras)"""

    model = keras.Sequential()

    model.add(layers.Conv2D(6, (5, 5), strides=1, input_shape=(28, 28, 1), activation='relu'))
    model.add(layers.MaxPool2D((2, 2), strides=2))

    model.add(layers.Conv2D(16, (5, 5), strides=1, activation='relu'))
    model.add(layers.MaxPool2D((2, 2), strides=2))

    model.add(layers.Flatten())

    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


class ModelMetrics(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def on_epoch_end(self, batch, logs={}):
        y_pred = self.model.predict(X_test)

        _precision, _recall, _f1, _sample = score(y_test, y_pred)

        self.precisions.append(_precision)
        self.recalls.append(_recall)
        self.f1_scores.append(_f1)


X_train0, y_train0, X_train1, y_train1 = dataset_classify(X_train, y_train)
X_test0, y_test0, X_test1, y_test1 = dataset_classify(X_test, y_test)

X_train0 = X_train0.reshape([-1, 28, 28, 1]).astype('float32')
X_train1 = X_train1.reshape([-1, 28, 28, 1]).astype('float32')
X_test0 = X_test0.reshape([-1, 28, 28, 1]).astype('float32')
X_test1 = X_test1.reshape([-1, 28, 28, 1]).astype('float32')

y_train0 = to_categorical(y_train0)
y_train1 = to_categorical(y_train1)
y_test0 = to_categorical(y_test0)
y_test1 = to_categorical(y_test1)

metrics2 = ModelMetrics()
model2 = le_net5()
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(
    X_train1,
    y_train1,
    validation_split=0.1,
    epochs=15,
    batch_size=32,
    callbacks=[metrics2]
)

metrics1 = ModelMetrics()
model1 = le_net5()
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model1.fit(
    X_train0,
    y_train0,
    validation_split=0.1,
    epochs=15,
    batch_size=32,
    callbacks=[metrics1]
)
