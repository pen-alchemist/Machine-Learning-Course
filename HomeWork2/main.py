import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from joblib import dump, load
from matplotlib import pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def linear_regression_normal_equation(X, y):
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)

    try:
        theta = np.linalg.solve(X_transpose_X, X_transpose_y)
        return theta
    except np.linalg.LinAlgError:
        return None


if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent

    # Read csv file with pandas
    csv_path = f'{file_path}/kc_house_data.csv'
    df = pd.read_csv(csv_path)

    # Make basic data analysis with pandas
    df.head()
    print(f'df.shape\n {df.shape}')
    print(f'df.columns\n {df.columns}')
    print(f'df.info()\n {df.info()}')
    print(f'df.describe()\n {df.describe()}')
    print(f'df.sort_values\n {df.sort_values(by="sqft_living", ascending=True)}')
    print(f'df[\'sqft_living\'].mean()\n {df["sqft_living"].mean()}')
    print(f'df[\'sqft_living\'].min()\n {df["sqft_living"].min()}')
    print(f'df[\'sqft_living\'].max()\n {df["sqft_living"].max()}')
    print(f'df[\'sqft_living\'].skew()\n {df["sqft_living"].skew()}')
    comp_df = df[df['sqft_living'] > 10000]
    print(f'Comparation df "sqft_living" with 10000\n {comp_df}')
    print(f'df[[\'yr_built\', \'yr_renovated\']].median()\n '
          f'{df[["yr_built", "yr_renovated"]].median()}')

    # Select columns from dataset that can be used in multivariate linear regression
    multi_selection = df.drop(['id', 'date', 'price', 'zipcode'], axis=1)
    print(f'multi_selection.head(101)\n {multi_selection.head(101)}')
    X = np.asarray(df['sqft_living'].values.tolist())
    # Reshaping the Dependent features
    X = X.reshape(len(X), 1)  # Changing the shape from (50,) to (50,1)\
    print(f'X is:\n {X}')
    print(f'X.shape\n {X.shape}')
    print(f'X.max() - X.min()\n {X.max() - X.min()}')
    y = np.asarray(df['price'].values.tolist())
    # Reshaping the Dependent features
    y = y.reshape(len(y), 1)  # Changing the shape from (50,) to (50,1)
    print(f'y is:\n {y}')
    print(f'y.shape\n {y.shape}')
    plt.plot(X, y, color='green', marker='o', linestyle='solid')
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    plt.savefig(f'{file_path}/graph1.png')
    plt.savefig(f'{file_path}/graph1.pdf')
    plt.close()

    # Let's create a DataFrame "Independent_Variables" to visualize our final independent features
    independ_variables = pd.DataFrame(X)
    print(f'Independent variables:\n {independ_variables}')

    # Univariate model training
    random_state = 42
    X, y = make_regression(random_state=random_state, n_features=1, noise=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state)

    model = LinearRegression()
    print(f'Model:\n {model}')
    model.fit(X_train, y_train)
    print(f'model.fit(X_train, y_train)\n {model.fit(X_train, y_train)}')

    scalar = StandardScaler(with_std=True)
    scalar.fit(X_test, y_test)
    to_compare_metrics = scalar.scale_
    print(f'To_compare_metrics (scalar.scale_)\n {to_compare_metrics}')

    model.predict(X_test)
    print(f'model.predict(X_test)\n {model.predict(X_test)}')
    model.score(X_test, y_test)
    print(f'model.score(X_test, y_test)\n {model.score(X_test, y_test)}')

    # Save model with pickle
    with open(f'{file_path}/modelmultivar.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save model with joblib
    dump(model, f'{file_path}/modelunivar.joblib')
    # Save data to test
    dump([X, y], f'{file_path}/test_data_univar.joblib')

    # Multivariate model training
    multi_selection = df.drop(['id', 'date', 'price', 'zipcode'], axis=1)
    print(f'multi_selection (X):\n {multi_selection}')
    X = np.asarray(multi_selection.values.tolist())
    print(f'X is:\n {X}')
    print(f'X.shape\n {X.shape}')
    y = np.asarray(df['price'].values.tolist())
    print(f'y is:\n {y}')
    print(f'y.shape\n {y.shape}')
    y = y.reshape(len(y), 1)
    print(f'y.shape\n {y.shape}')
    X_normalized = normalize(X, norm="l1")
    print(f'X_normalized\n {X_normalized}')
    print(X_normalized)
    print(f'X_normalized.shape\n {X_normalized.shape}')
     
    # Add a column of ones to X for the intercept term
    X_with_intercept1 = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
     
    theta1 = linear_regression_normal_equation(X_with_intercept1, y)
    if theta1 is not None:
        print(f'Theta is: \n{theta1}')
    else:
        print("Unable to compute theta. The matrix X_transpose_X is singular.")

    loaded_test_data = load(f'{file_path}/test_data_univar.joblib')
    loaded_X = loaded_test_data[0]
    loaded_y = loaded_test_data[1]

    # Add a column of ones to X for the intercept term
    X_with_intercept2 = np.c_[np.ones((loaded_X.shape[0], 1)), loaded_X]

    theta2 = linear_regression_normal_equation(X_with_intercept2, loaded_y)
    if theta2 is not None:
        print(f'Theta is: \n{theta2}')
    else:
        print("Unable to compute theta. The matrix X_transpose_X is singular.")

    plt.plot(X_normalized, y, color='green', marker='o', linestyle='solid')
    plt.plot(loaded_X, loaded_y, color='red', marker='o', linestyle='solid')
    plt.xlabel(f'Multivar Linear Regression Model, theta: {theta1}')
    plt.ylabel(f'Univar Linear Regression Model, theta: {theta2}')
    plt.title('Comparison of the models (multiple and univar)')
    plt.savefig(f'{file_path}/graph_comparing1.png')
    plt.close()

    # Create data set.
    random_state = 101
    X_normalized, y = make_regression(random_state=random_state, n_features=17, noise=17)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X_normalized, y, test_size=0.3, random_state=random_state)

    model = LinearRegression()
    print(f'Model:\n {model}')
    model.fit(X_train1, y_train1)
    print(f'model.fit(X_train, y_train)\n {model.fit(X_train1, y_train1)}')

    scalar = StandardScaler(with_std=True)
    scalar.fit(X_test1, y_test1)
    to_compare_metrics = scalar.scale_
    print(f'To_compare_metrics (scalar.scale_)\n {to_compare_metrics}')

    model.predict(X_test1)
    print(model.predict(X_test1))
    model.score(X_test1, y_test1)
    print(f'model.score(X_test, y_test)\n {model.score(X_test1, y_test1)}')

    # Save model with pickle
    with open('../modelmultivar.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Save model with joblib
    dump(model, f'{file_path}/modelmultivar.joblib')

    loaded_multimodel = load(f'{file_path}/modelmultivar.joblib')
    loaded_unimodel = load(f'{file_path}/modelunivar.joblib')
    loaded_testdata = load(f'{file_path}/test_data_univar.joblib')

    loaded_multimodel.predict(X_test1)
    multi_predict = loaded_multimodel.predict(X_test1)

    print(f'multivar_predict:\n {multi_predict}')

    plt.scatter(multi_predict, y_test1, color='red', marker='o')
    plt.savefig(f'{file_path}/graph2.png')
    plt.close()

    loaded_testdata[0], loaded_testdata[1] = make_regression(
        random_state=random_state, n_features=1, noise=1)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        loaded_testdata[0], loaded_testdata[1], test_size=0.4, random_state=random_state)

    loaded_unimodel.predict(X_test2)
    univar_predict = loaded_unimodel.predict(X_test2)

    print(f'univar_predict:\n {univar_predict}')

    plt.scatter(univar_predict, y_test2, color='green', marker='o')
    plt.savefig(f'{file_path}/graph3.png')
    plt.close()

    # Creating two multiple linear regression models:
    csv_path = f'{file_path}/customers-100.csv'
    df = pd.read_csv(csv_path)

    df.head()
    multi_selection = df.drop(
        [
            'Salary',
            'Vacation',
            'Index',
            'Customer Id',
            'First Name',
            'Last Name',
            'Phone 1',
            'Phone 2',
            'Email'
        ],
        axis=1
    )
    X = np.asarray(multi_selection.values.tolist())
    print(f'X is:\n {X}')
    print(f'X.shape\n {X.shape}')
    y = np.asarray(df['Salary'].values.tolist())
    y = y.reshape(len(y), 1)
    print(f'y is:\n {y}')
    print(f'y.shape\n {y.shape}')

    # Add a column of ones to X for the intercept term
    X_with_intercept3 = np.c_[np.ones((X.shape[0], 1)), X]

    theta1 = linear_regression_normal_equation(X_with_intercept3, y)
    if theta1 is not None:
        print(f'Theta is: \n{theta1}')
    else:
        print("Unable to compute theta. The matrix X_transpose_X is singular.")

    random_state = 101
    X, y = make_regression(random_state=random_state, n_features=17, noise=17)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state)

    model = LinearRegression()
    print(f'Model:\n {model}')
    model.fit(X_train, y_train)
    print(f'model.fit(X_train, y_train)\n {model.fit(X_train, y_train)}')

    scalar = StandardScaler(with_std=True)
    scalar.fit(X_test, y_test)
    to_compare_metrics = scalar.scale_
    print(f'To_compare_metrics (scalar.scale_)\n {to_compare_metrics}')

    model.predict(X_test)
    prediction = model.predict(X_test)
    print(f'Prediction:\n {prediction}')

    plt.scatter(prediction, y_test, color='yellow', marker='o')
    plt.savefig(f'{file_path}/graph4.png')
    plt.close()

    model.score(X_test, y_test)
    print(f'model.score(X_test, y_test)\n {model.score(X_test, y_test)}')
    model.coef_
    print(f'model.coef_:\n {model.coef_}')
    model.mean_
    print(f'model.mean_:\n {model.mean_}')
    model.var_
    print(f'model.var_:\n {model.var_}')
    dump(model, f'{file_path}/multivarmodel1.joblib')

    y = np.asarray(df['Vacation'].values.tolist())
    y = y.reshape(len(y), 1)
    print(f'y is:\n {y}')
    print(f'y.shape\n {y.shape}')

    # Add a column of ones to X for the intercept term
    X_with_intercept4 = np.c_[np.ones((X.shape[0], 1)), X]

    theta2 = linear_regression_normal_equation(X_with_intercept4, y)
    if theta2 is not None:
        print(f'Theta is: \n{theta2}')
    else:
        print("Unable to compute theta. The matrix X_transpose_X is singular.")

    random_state = 101
    X, y = make_regression(random_state=random_state, n_features=17, noise=17)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state)

    model = LinearRegression()
    print(f'Model:\n {model}')
    model.fit(X_train, y_train)
    print(f'model.fit(X_train, y_train)\n {model.fit(X_train, y_train)}')

    scalar = StandardScaler(with_std=True)
    scalar.fit(X_test, y_test)
    to_compare_metrics = scalar.scale_
    print(f'To_compare_metrics (scalar.scale_)\n {to_compare_metrics}')

    model.predict(X_test)
    prediction = model.predict(X_test)
    print(f'Prediction:\n {prediction}')

    plt.scatter(prediction, y_test, color='blue', marker='o')
    plt.savefig(f'{file_path}/graph5.png')
    plt.close()

    model.score(X_test, y_test)
    print(f'model.score(X_test, y_test)\n {model.score(X_test, y_test)}')
    model.coef_
    print(f'model.coef_:\n {model.coef_}')
    model.mean_
    print(f'model.mean_:\n {model.mean_}')
    model.var_
    print(f'model.var_:\n {model.var_}')
    dump(model, f'{file_path}/multivarmodel2.joblib')

    model1 = load(f'{file_path}/multivarmodel1.joblib').predict(X_test)
    model2 = load(f'{file_path}/multivarmodel2.joblib').predict(X_test)

    plt.plot(model1, model2, color='blue', marker='o')
    plt.savefig(f'{file_path}/graph6.png')
    plt.close()
