import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Dropout

from sklearn.model_selection import train_test_split

# Read csv file with pandas
csv_path = f'{os.getcwd()}/kc_house_data.csv'
df = pd.read_csv(csv_path)

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

random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=random_state)

print(f'Training Data: {X_train.shape}')
print(f'Training Labels: {y_train.shape}')
print(f'Labels: {y_train}')

model = Sequential()
dropout = Dropout(0.2)
model.add(dropout)
hidden_layer1 = Dense(units=1, activation="sigmoid", name="hidden_layer1")
model.add(hidden_layer1)

model.compile(loss='mean_squared_error', optimizer="adam", metrics=['accuracy'])

epochs = 10
model.fit(X_train, y_train, epochs=epochs, shuffle=True, batch_size=128, verbose=2)
