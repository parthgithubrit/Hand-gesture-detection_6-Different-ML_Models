from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras import layers, Sequential
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import os
import numpy as np
DATA_PATH = os.path.join('/content/drive/MyDrive/MP_Data')

actions = np.array(['A','B','C'])

no_sequences = 15

sequence_length = 15

label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))

            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
X= np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(15,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200
          , callbacks=[tb_callback])




model.summary()
model_json = model.to_json()
with open("/content/drive/MyDrive/model.json", "w") as json_file:
    json_file.write(model_json)
model.save('/content/drive/MyDrive/model.h5')
#CNN
#Reshape data for CNN
X_train = X_train.reshape(-1, sequence_length, 63, 1)
X_test = X_test.reshape(-1, sequence_length, 63, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(sequence_length, 63, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model.fit(X_train, y_train, epochs=200, verbose=1)

model.summary()
model_json = model.to_json()
with open("/content/drive/MyDrive/modelcnnmp.json", "w") as json_file:
    json_file.write(model_json)
model.save('/content/drive/MyDrive/modelcnnmp.h5')

# Reshape data for FNN
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
model = Sequential()
model.add(Flatten(input_shape=(X_train.shape[1:])))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=1)

model.summary()
model_json = model.to_json()
with open("/content/drive/MyDrive/modelfnn.json", "w") as json_file:
    json_file.write(model_json)
model.save('/content/drive/MyDrive/modelfnn.h5')