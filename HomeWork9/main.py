import os 
import keras
import torch
import numpy as np
import pandas as pd

import tensorflow as tf

from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


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
        self.score = []
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        y_pred = self.model.predict(X_test)
        
        score_eval, acc_eval = self.model.evaluate(
            X_test, y_test, 
            batch_size=32
        )

        self.score.append(score_eval)
        self.acc.append(acc_eval)


X_train = X_train.reshape([-1, 28, 28, 1]).astype('float32')
X_test = X_test.reshape([-1, 28, 28, 1]).astype('float32')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


metrics = ModelMetrics()
model = le_net5()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=32,
    callbacks=[metrics]
)


def path_to_dataset():
    # Read csv file with pandas
    dataset_path = f'{os.getcwd()}/dataset'

    return dataset_path


def csv_reader(dataset_path):
    # Read csv file with pandas
    csv_path = f'{dataset_path}/bald_people.csv'
    df = pd.read_csv(csv_path)

    return df


def csv_to_X(input_df, dataset_path):
    X = []
    image_list = input_df['images']

    for image in image_list:
        image_path = f'{dataset_path}/{image}'
        img = load_img(image_path, target_size=(28, 28))
        img = img_to_array(img)

        X.append(img)

    X = np.array(X)

    return X


dataset_path = path_to_dataset()
df = csv_reader(dataset_path)

X = csv_to_X(df, dataset_path)

y = np.array(df['type'].values.tolist())


random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=random_state)


datagen = ImageDataGenerator(rescale=1.0/255.0)
train_iterator = datagen.flow(X_train, y_train, batch_size=32)
test_iterator = datagen.flow(X_test, y_test, batch_size=32)

X_batch_train, y_batch_train = next(train_iterator)
X_batch_train = X_batch_train.reshape([-1, 28, 28, 1]).astype('float32')
y_batch_train = y_batch_train.reshape(-1, 1)

lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)


model.trainable = False
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(120, activation='relu'))

model.summary()

metrics = ModelMetrics()
model.fit(
    X_batch_train,
    y_batch_train,
    epochs=15,
    batch_size=32,
    callbacks=[metrics]
)
