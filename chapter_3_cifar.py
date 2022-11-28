from tensorflow import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.losses import CategoricalCrossentropy
import numpy as np
import matplotlib.pyplot as plt

# Step 1
# Load the Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fig = plt.figure(figsize=(20, 5))
for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))

# Step 2
# Rescale the images
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# One Hot encoding
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# split the training dataset for training and validation
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

print('x_train shape', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'validation samples')


#Step 3 Define the model architecture
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Dropout layer to avoid overfitting with a 30% rate
model.add(Dropout(.3))

# flatten the last feature map into a vector of features
model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dropout(.4))

model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss=CategoricalCrossentropy(), optimizer='rmsprop', metrics=['accuracy'])
#Step 5 Train the model
checkpointer = ModelCheckpoint(filepath='checkpoints/model.weights.best.hdf5', verbose=1, save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=100,
                 validation_data=(x_valid, y_valid), callbacks=[checkpointer],
                 verbose=True, shuffle=True)

# Step 6 load the model with the best val_acc
model.load_weights('model.weights.best.hdf5')

# Step 7 evalaute the model
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])