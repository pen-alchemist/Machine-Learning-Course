from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

random_state = 42
X, y = make_classification(random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=random_state)

print(f'Training Data: {X_train.shape}')
print(f'Training Labels: {y_train.shape}')
print(f'Labels: {y_train}')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.transform(y_test)

epochs = 10

model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))  # input layer requires input_dim param
model.add(Dense(16, activation='softmax'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

tensorboard_callback = TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    update_freq="epoch",
)

model.fit(
    X_train,
    y_train,
    epochs=epochs,
    shuffle=True,
    batch_size=128,
    verbose=2,
    callbacks=[tensorboard_callback]
)

es = EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=1,
    verbose=1,
    mode='auto'
)
model.fit(
    X_train,
    y_train,
    epochs=epochs,
    shuffle=True,
    batch_size=128,
    verbose=2,
    callbacks=[es]
)

scores = model.evaluate(X_test, y_test)
print(model.metrics_names[0], model.metrics_names[1])

checkpoint_filepath = 'checkpoint.model.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Model is saved at the end of every epoch, if it's the best seen so far.
model.fit(
    X_train,
    y_train,
    epochs=epochs,
    shuffle=True,
    batch_size=128,
    verbose=2,
    callbacks=[model_checkpoint_callback]
)
model.save(checkpoint_filepath)

# The model (that are considered the best) can be loaded as -
load_model(checkpoint_filepath)
