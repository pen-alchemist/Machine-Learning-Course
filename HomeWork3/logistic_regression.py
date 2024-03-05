from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


X, y = make_classification(random_state=42)
# Data normalization
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=35)

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
accuracy = accuracy_score(y_test, predict)
conf_matrix = confusion_matrix(y_test, predict)
cls_report = classification_report(y_test, predict)
loss = log_loss(y_test, predict_proba, labels=None)


if __name__ == "__main__":
    print('================================================')
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
