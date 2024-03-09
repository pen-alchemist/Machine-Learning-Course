import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# Dataset
X, y = make_classification(random_state=42)
# Data normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)

# Model making
model1 = LogisticRegression(random_state=15, max_iter=1000)
model1.fit(X_train, y_train)
predict = model1.predict(X_test)
score = model1.score(X_test, y_test)
predict_proba = model1.predict_proba(X_test)
MAE_logistic = mean_absolute_error(y_test, predict)
RMSE_logistic = np.sqrt(mean_squared_error(y_test, predict))

print(predict)
print(score)
print(predict_proba)


# CV Data normalization
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.4, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

# Selecting hyper parameter
poly_converter = PolynomialFeatures(include_bias=False)
poly_features_train = poly_converter.fit_transform(X_train)
poly_features_eval = poly_converter.fit_transform(X_eval)

ridge_model = Ridge(alpha=1)
ridge_model.fit(poly_features_train, y_train)
predict1 = ridge_model.predict(poly_features_eval)
RMSE = np.sqrt(mean_squared_error(y_eval, predict1))
print('Ridge RMSE: ', RMSE)

lasso_model = Lasso(alpha=1)
lasso_model.fit(poly_features_train, y_train)
predict2 = lasso_model.predict(poly_features_eval)
RMSE = np.sqrt(mean_squared_error(y_eval, predict2))
print('Lasso RMSE: ', RMSE)

elastic_model_pre = ElasticNet(alpha=1)
elastic_model_pre.fit(poly_features_train, y_train)
predict3 = elastic_model_pre.predict(poly_features_eval)
RMSE = np.sqrt(mean_squared_error(y_eval, predict3))
print('ElasticNet RMSE: ', RMSE)


# ElasticNet is best model in this case
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
elastic_model_result = ElasticNet(alpha=1)
elastic_model_result.fit(X_train, y_train)

predict_elastic = elastic_model_result.predict(X_test)
score_elastic = elastic_model_result.score(X_test, y_test)
MAE_elastic = mean_absolute_error(y_test, predict_elastic)
RMSE_elastic = np.sqrt(mean_squared_error(y_test, predict_elastic))


# Comparing with model1
print('ElasticNet SCORE: ', score_elastic, '\n')
print('LogisticRegression SCORE: ', score, '\n')
print('ElasticNet MAE: ', MAE_elastic, '\n')
print('LogisticRegression MAE: ', MAE_logistic, '\n')
print('ElasticNet RMSE: ', RMSE_elastic, '\n')
print('LogisticRegression RMSE: ', RMSE_logistic, '\n')
print('ElasticNet Loss: ', RMSE_elastic, '\n')
print('LogisticRegression Loss: ', RMSE_logistic, '\n')
