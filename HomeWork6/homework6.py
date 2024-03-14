import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential

from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'Training Data: {x_train.shape}')
print(f'Training Labels: {y_train.shape}')
print(f'Labels: {y_train}')


def dataset_to_dict(labels, features):
    result_dict = dict()
    labels = np.ndarray.tolist(labels)
    features = np.ndarray.tolist(features)
    
    for i in range(len(labels)):
        result_dict[labels[i]] = features[i]
        
    return result_dict


def dataset_selecting(data_dict):
    x_train, y_train = list(), list()
    
    for label, feature in data_dict.items():
        if label >= 5 or label < 5:
            y_train.append(label)
            x_train.append(feature)

    y_train = np.array(y_train)
    x_train = np.array(x_train)
            
    return x_train, y_train


data_dict = dataset_to_dict(y_train, x_train)
x_train, y_train =  dataset_selecting(data_dict)

x_train = x_train.flatten()
x_test = x_test.flatten()

x_train = x_train.reshape(len(x_train), 1)
x_test = x_test.reshape(len(x_test), 1)

x_train = normalize(x_train)
x_test = normalize(x_test)

# confirm scale of pixels
print(f'Train min={x_train.min()} max={x_train.max()}')
print(f'Test min={x_test.min()} max={x_test.max()}')

lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.transform(y_test)

# Model making
model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(x_train, y_train)
predict = model.predict(x_test)

print(f'Predict of logistic regression is: {predict}')


# PCA component or unit matrix
u = eigenvectors[:,:n_components]
pca_component = pd.DataFrame(u,
                             index = cancer['feature_names'],
                             columns = ['PC1','PC2']
                            )
 
# plotting heatmap
plt.figure(figsize =(5, 7))
sns.heatmap(pca_component)
plt.title('PCA Component')
plt.show()
