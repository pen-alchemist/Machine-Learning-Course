import pickle
import numpy as np
import pandas as pd

from joblib import dump, load
from matplotlib import pyplot as plt

from sklearn import svm
from sklearn import datasets
from sklearn.metrics import det_curve
from sklearn.preprocessing import normalize
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Read csv file with pandas
    csv_path = 'kc_house_data.csv'
    df = pd.read_csv(csv_path)

    # Make basic data analysis with pandas
    df.head()
    print(df.shape)
    print(df.columns)
    print(df.info())
    print(df.describe())
    print(df.sort_values(by='sqft_living', ascending=True))
    print(df['sqft_living'].mean())
    print(df['sqft_living'].min())
    print(df['sqft_living'].max())
    print(df['sqft_living'].skew())
    comp_df = df[df['sqft_living'] > 10000]
    print(comp_df)
    print(df[['yr_built', 'yr_renovated']].median())

    # Select columns from dataset that can be used in multivariate linear regression
    multi_selection = df.drop(['id', 'date', 'price', 'zipcode'], axis=1)
    print(multi_selection.head(101))
    X = np.asarray(df['sqft_living'].values.tolist())
    # Reshaping the Dependent features
    X = X.reshape(len(X), 1)  # Changing the shape from (50,) to (50,1)\
    print(X)
    print(X.shape)
    print(X.max() - X.min())
    y = np.asarray(df['price'].values.tolist())
    # Reshaping the Dependent features
    y = y.reshape(len(y), 1)  # Changing the shape from (50,) to (50,1)
    print(y)
    print(y.shape)
    plt.plot(X, y, color='green', marker='o', linestyle='solid')
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    plt.savefig('graph1.png')
    plt.savefig('graph1.pdf')

    # Let's create a DataFrame "Independent_Variables" to visualize our final independent features
    independ_variables = pd.DataFrame(X)
    print(independ_variables)

    # Univariate model training
    random_state = 42
    X, y = make_regression(random_state=random_state, n_features=1, noise=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state)

    model = LinearRegression()
    print(model)
    model.fit(X_train, y_train)
    print(model.fit(X_train, y_train))

    model.predict(X_test)
    print(model.predict(X_test))
    model.score(X_test, y_test)
    print(model.score(X_test, y_test))

    # Save model with pickle
    with open('modelmultivar.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save model with joblib
    dump(model, 'modelunivar.joblib')
    # Save data to test
    dump([X_test, y_test], 'test_data_univar.joblib')

    # Multivariate model training
    multi_selection = df.drop(['id', 'date', 'price', 'zipcode'], axis=1)
    print(multi_selection)
    X = np.asarray(multi_selection.values.tolist())
    print(X)
    print(X.shape)
    y = np.asarray(df['price'].values.tolist())
    print(y)
    print(y.shape)
    y = y.reshape(len(y), 1)
    print(y.shape)
    X_normalized = normalize(X, norm="l1")
    print(X_normalized)
    print(X_normalized.shape)

    random_state = 101

    X_normalized, y = make_regression(random_state=random_state, n_features=17, noise=17)
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.3, random_state=random_state)

    model = LinearRegression()
    print(model)
    model.fit(X_train, y_train)
    print(model.fit(X_train, y_train))

    model.predict(X_test)
    print(model.predict(X_test))
    model.score(X_test, y_test)
    print(model.score(X_test, y_test))

    # Save model with pickle
    with open('modelmultivar.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save model with joblib
    dump(model, 'modelmultivar.joblib')

    loaded_multimodel = load('modelmultivar.joblib')
    loaded_unimodel = load('modelunivar.joblib')
    loaded_testdata = load('test_data_univar.joblib')

    loaded_multimodel.predict(X_test)
    multi_predict = loaded_multimodel.predict(X_test)

    print(multi_predict)

    plt.scatter(multi_predict, y_test, color='red', marker='o')
    plt.savefig('graph2.png')

    loaded_unimodel.predict(loaded_testdata[0])
    univar_predict = loaded_unimodel.predict(loaded_testdata[0])

    print(univar_predict)

    plt.scatter(univar_predict, loaded_testdata[1], color='green', marker='o')
    plt.savefig('graph3.png')

    # Creating two multiple linear regression models:
    csv_path = 'customers-100.csv'
    df = pd.read_csv(csv_path)

    df.head()
    multi_selection = df.drop(
        ['Salary', 'Vacation',
         'Index', 'Customer Id',
         'First Name', 'Last Name',
         'Phone 1', 'Phone 2'],
        axis=1
    )
    X = np.asarray(multi_selection.values.tolist())
    print(X)
    print(X.shape)
    y = np.asarray(df['Salary'].values.tolist())
    y = y.reshape(len(y), 1)
    print(y)
    print(y.shape)

    random_state = 101
    X, y = make_regression(random_state=random_state, n_features=17, noise=17)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state)

    model = LinearRegression()
    print(model)
    model.fit(X_train, y_train)
    print(model.fit(X_train, y_train))

    model.predict(X_test)
    prediction = model.predict(X_test)
    print(prediction)

    plt.scatter(prediction, y_test, color='yellow', marker='o')
    plt.savefig('graph4.png')

    model.score(X_test, y_test)
    print(model.score(X_test, y_test))
    model.coef_
    print(model.coef_)
    dump(model, 'multivarmodel1.joblib')

    y = np.asarray(df['Vacation'].values.tolist())
    y = y.reshape(len(y), 1)
    print(y)
    print(y.shape)

    random_state = 101
    X, y = make_regression(random_state=random_state, n_features=17, noise=17)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state)

    model.predict(X_test)
    prediction = model.predict(X_test)
    print(prediction)

    plt.scatter(prediction, y_test, color='blue', marker='o')
    plt.savefig('graph5.png')

    model.score(X_test, y_test)
    print(model.score(X_test, y_test))
    model.coef_
    print(model.coef_)
    dump(model, 'multivarmodel2.joblib')

    model1 = load('multivarmodel1.joblib').predict(X_test)
    model2 = load('multivarmodel2.joblib').predict(X_test)

    plt.plot(model1, model2, color='blue', marker='o')
    plt.savefig('graph6.png')
