import numpy as np

from keras.datasets import mnist
from keras.datasets import cifar10

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


def dataset_to_dict(labels, features):
    """Returns dict with labels and features"""
    result_dict = dict()
    labels = np.ndarray.tolist(labels)
    features = np.ndarray.tolist(features)
    
    for i in range(len(labels)):
        result_dict[labels[i]] = features[i]
        
    return result_dict


def dataset_selecting(data_dict):
    """Returns dict with x_train and y_train if y >= 5 or y < 5"""
    x_train, y_train = list(), list()
    
    for label, feature in data_dict.items():
        if label >= 5 or label < 5:
            y_train.append(label)
            x_train.append(feature)

    y_train = np.array(y_train)
    x_train = np.array(x_train)
            
    return x_train, y_train


# Load data MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'Training Data: {x_train.shape}')
print(f'Training Labels: {y_train.shape}')
print(f'Labels: {y_train}')

data_dict = dataset_to_dict(y_train, x_train)
x_train, y_train = dataset_selecting(data_dict)

x_train = x_train.flatten()
x_test = x_test.flatten()

x_train = x_train.reshape(len(x_train), 1)
x_test = x_test.reshape(len(x_test), 1)

# Normalizing data to 0,1
x_train = normalize(x_train)
x_test = normalize(x_test)

# Confirm scale of pixels
print(f'Train min={x_train.min()} max={x_train.max()}')
print(f'Test min={x_test.min()} max={x_test.max()}')

# Encoding labels
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.transform(y_test)

# Model making
model = LogisticRegression(random_state=42, max_iter=100000)
model.fit(x_train, y_train)
predict = model.predict(x_test)

print(f'Predict of logistic regression is: {predict}')


# Load data CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f'Training Data: {x_train.shape}')
print(f'Training Labels: {y_train.shape}')
print(f'Labels: {y_train}')

data_dict = dataset_to_dict(y_train, x_train)
x_train, y_train = dataset_selecting(data_dict)

# Let's say, components = 2
pca = PCA(n_components=2)
pca.fit(x_train)
x_pca_train = pca.transform(x_train)
x_pca_test = pca.transform(x_test)

# Normalizing data to 0,1
x_train = normalize(x_train)
x_test = normalize(x_test)

# Confirm scale of pixels
print(f'Train min={x_train.min()} max={x_train.max()}')
print(f'Test min={x_test.min()} max={x_test.max()}')

# Encoding labels
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.transform(y_test)

# Model making
model = LogisticRegression(random_state=42, max_iter=100000)
model.fit(x_train, y_train)
predict = model.predict(x_test)

print(f'Predict of logistic regression is: {predict}')
