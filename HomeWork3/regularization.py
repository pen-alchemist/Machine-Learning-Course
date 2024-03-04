import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.datasets import make_regression


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Ridge
ridge_model = Ridge(alpha=0.001, max_iter=1000)
ridge_model.fit(X_train, y_train)
predict = ridge_model.predict(X_test)

print('Price column mean value: ', df['Salary'].mean())
MAE = mean_absolute_error(y_test, predict)
RMSE = np.sqrt(mean_squared_error(y_test, predict))

# Lasso
lasso_model = Lasso(alpha=0.001, max_iter=1000, tol=0.1)
lasso_model.fit(X_train, y_train)
predict_2 = lasso_model.predict(X_test)

print('Price column mean value: ', df['Salary'].mean())
MAE = mean_absolute_error(y_test, predict_2)
print('MAE: ', MAE)
RMSE = np.sqrt(mean_squared_error(y_test, predict_2))
print('RMSE: ', RMSE, '\n')


# ElasticNet
elastic_model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=1000, tol=0.1)
elastic_model.fit(X_train, y_train)
predict_3 = elastic_model.predict(X_test)

print('Price column mean value: ', df['Salary'].mean())
MAE = mean_absolute_error(y_test, predict_3)
print('Mean absolute error: ', MAE)
RMSE = np.sqrt(mean_squared_error(y_test, predict_3))
print('Mean Squared Error: ', RMSE, '\n')


# Making y_true and y_pred to calculate loss
lab_enc = preprocessing.LabelEncoder()
y_true = lab_enc.fit_transform(y_test)
y_pred1 = lab_enc.fit_transform(predict)
y_pred2 = lab_enc.fit_transform(predict_2)
y_pred3 = lab_enc.fit_transform(predict_3)

lb = preprocessing.LabelBinarizer()
y_true = lb.fit_transform(y_true)
y_pred1 = lb.transform(y_pred1)
y_pred2 = lb.transform(y_pred2)
y_pred3 = lb.transform(y_pred3)


# Comparing regulization models losses
print(f'Predict of first model Ridge: \n {log_loss(y_true, y_pred1, labels=None)}')
print(f'Predict of second model Lasso: \n {log_loss(y_true, y_pred2, labels=None)}')
print(f'Predict of third model ElasticNet: \n {log_loss(y_true, y_pred3, labels=None)}')

# Comparing regulization models predicts
print(f'Predict of first model Ridge: \n {predict}')
print(f'Predict of second model Lasso: \n {predict_2}')
print(f'Predict of third model ElasticNet: \n {predict_3}')

# Comparing regulization models scores
print(f'Score of first model Ridge: \n {ridge_model.score(X_test, y_test)}')
print(f'Score of second model Lasso: \n {lasso_model.score(X_test, y_test)}')
print(f'Score of third model ElasticNet: \n {elastic_model.score(X_test, y_test)}')

# Save the best model in a list
dump(elastic_model, "best_model.joblib")


# CV
# Data normalization
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.4, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)
X_test = scaler.transform(X_test)

eval_rmse_errors = []

# Selecting hyper parameter
d_range = np.asarray([1, 2, 3, 4])
for d in d_range:
    poly_converter = PolynomialFeatures(degree=d, include_bias=False)
    poly_features_train = poly_converter.fit_transform(X_train)
    poly_features_eval = poly_converter.fit_transform(X_eval)

    model = Ridge(alpha=1)
    model.fit(poly_features_train, y_train)
    pred = model.predict(poly_features_eval)

    RMSE = np.sqrt(mean_squared_error(y_eval, pred))

    eval_rmse_errors.append(RMSE)

print(eval_rmse_errors, '\n')


optimal_d = d_range[np.argmin(np.array(eval_rmse_errors))]
poly_converter_2 = PolynomialFeatures(degree=optimal_d, include_bias=False)
poly_features_train_2 = poly_converter_2.fit_transform(X_train)
poly_features_test_2 = poly_converter_2.fit_transform(X_test)
final_model = Ridge(alpha=1)
final_model.fit(poly_features_train_2, y_train)
final_predict = final_model.predict(poly_features_test_2)
RMSE = np.sqrt(mean_squared_error(y_test, final_predict))
print('Mean Squared Error: ', RMSE, '\n')


# Making y_true and y_pred to calculate loss
lab_enc = preprocessing.LabelEncoder()
y_true = lab_enc.fit_transform(y_test)
y_pred = lab_enc.fit_transform(final_predict)

lb = preprocessing.LabelBinarizer()
y_true = lb.fit_transform(y_true)
y_pred = lb.transform(y_pred)

log_loss(y_true, y_pred, labels=None)
