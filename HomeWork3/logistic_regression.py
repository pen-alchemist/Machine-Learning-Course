import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Reading file and dropping values
csv_path = 'customers-100.csv'
df = pd.read_csv(csv_path, quoting=csv.QUOTE_NONNUMERIC)
drop_list = [
    'Salary', 
    'Vacation', 
    'Index', 
    'Customer Id', 
    'First Name', 
    'Last Name', 
    'Phone 1', 
    'Email', 
    'Phone 2'
]
multi_selection = df.drop(drop_list, axis=1)


# Making variables
X = np.asarray(multi_selection.values.tolist())
y = np.asarray(df['Salary'].values.tolist()) 
y = y.reshape(len(y), 1)


X, y = make_regression(random_state=4, n_features=17, noise=17)
# Data normalization
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=35)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler = OneHotEncoder(handle_unknown='ignore')
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)

# Model making
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
predict = model.predict(X_test)
score = model.score(X_test, y_test)
predict_proba = model.predict_proba(X_test)

# Making y_true and y_pred to calculate loss
y_pred = list(predict_proba)[0][:30]
y_pred = np.asarray(y_pred)
lab_enc = preprocessing.LabelEncoder()
y_pred = lab_enc.fit_transform(y_pred)
lb = preprocessing.LabelBinarizer()
y_true = lb.fit_transform(y_test)
y_pred = lb.transform(y_pred)
accuracy = accuracy_score(y_test, predict)
conf_matrix = confusion_matrix(y_test, predict)
cls_report = classification_report(y_test, predict)
loss = log_loss(y_true, y_pred, labels=None)


if __name__ == "__main__":
    print('================================================')
    print('# 1. Multi selection DF to make X variable')
    print(multi_selection)
    print('Model prediction - predict')
    print(predict)
    print('Model score')
    print(score)
    print('Model prediction - predict_proba')
    print(predict_proba)

    print('================================================')
    print('# 2. Stats of model')
    # Statisctics
    print(f'Accuracy:\n {accuracy}')
    print(f'Confusion_matrix:\n {conf_matrix}')
    print(f'Classification_report:\n {cls_report}')
    print(f'Loss:\n {loss}')
