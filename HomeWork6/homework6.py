import numpy as np

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.src.utils import to_categorical

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, LabelEncoder, StandardScaler


def dataset_selecting(features, labels):
    """Returns dict with x_train and y_train if y >= 5 or y < 5"""
    x, y = [], []

    for label_ind in range(len(labels)):
        label = labels[label_ind]

        if label >= 5 or label < 5:
            y.append(label)
            x.append(features[label_ind])

    x = np.asarray(x_train)
    y = np.asarray(x_train)

    return x, y


# Load data MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[0:99]
x_test = x_test[0:99]
y_train = y_train[0:99]
y_test = y_test[0:99]

print(f'Training Data: {x_train.shape}')
print(f'Training Labels: {y_train.shape}')
print(f'Labels: {y_train}')
print(f'Features: {x_train}')

x_train, y_train = dataset_selecting(x_train, y_train)
x_test, y_test = dataset_selecting(x_test, y_test)

print(f'Labels: {y_train}')
print(f'Features: {x_train}')

x_train = x_train.flatten()
x_test = x_test.flatten()
x_train = x_train.reshape(len(x_train), 1)
x_test = x_test.reshape(len(x_test), 1)
# Normalizing data to 0,1
x_train = normalize(x_train)
x_test = normalize(x_test)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# One hot encode labels
lab_encoder = LabelEncoder()
y_train = lab_encoder.fit_transform(y_train)
y_test = lab_encoder.transform(y_test)

# Confirm scale of pixels
print(f'Train min={x_train.min()} max={x_train.max()}')
print(f'Test min={x_test.min()} max={x_test.max()}')

y_test_orig = y_test
x_test_orig = x_test

# Model making
model1 = LogisticRegression(random_state=42, max_iter=10000)
model1.fit(x_train, y_train)
predict = model1.predict(x_test)

print(f'Predict of logistic regression is: {predict}')

# Load data CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[0:99]
x_test = x_test[0:99]
y_train = y_train[0:99]
y_test = y_test[0:99]

print(f'Training Data: {x_train.shape}')
print(f'Training Labels: {y_train.shape}')

x_train, y_train = dataset_selecting(x_train, y_train)
x_test, y_test = dataset_selecting(x_test, y_test)

print(f'Labels: {y_train}')
print(f'Features: {x_train}')

x_train = x_train.flatten()
x_test = x_test.flatten()
x_train = x_train.reshape(len(x_train), 1)
x_test = x_test.reshape(len(x_test), 1)

# Normalizing data to 0,1
x_train = normalize(x_train)
x_test = normalize(x_test)

# PCA, components = 2
pca = PCA(n_components=1)
x_pca_train = pca.fit_transform(x_train)
x_pca_test = pca.transform(x_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# One hot encode labels
lab_encoder = LabelEncoder()
y_train = lab_encoder.fit_transform(y_train)
y_test = lab_encoder.transform(y_test)

# Confirm scale of pixels
print(f'Train min={x_pca_train.min()} max={x_train.max()}')
print(f'Test min={x_pca_test.min()} max={x_test.max()}')

# Model making
model2 = LogisticRegression(random_state=42, max_iter=100000)
model2.fit(x_pca_train, y_train)
predict = model2.predict(x_pca_test)

print(f'Predict of logistic regression is: {predict}')

print(f'First model accuracy:{model1.score(x_test_orig, y_test_orig)}')
print(f'First model accuracy:{model2.score(x_test, y_test)}')
