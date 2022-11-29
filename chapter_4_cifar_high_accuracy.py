import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.losses import CategoricalCrossentropy
from keras import regularizers, optimizers
import numpy as np

from matplotlib import pyplot

# Step 2 Get the data ready for training
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

print('x_train = ', x_train.shape)
print('x_valid = ', x_valid.shape)
print('x_test = ', x_test.shape)

# Normalize the data
mean = np.mean(x_train, axis=(0,1,2,3))
std = np.std(x_train, axis=(0,1,2,3))
x_train = (x_train - mean) / (std + 1e-7)
x_valid = (x_valid - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)

# One-hot encoding labels
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_valid = np_utils.to_categorical(y_valid, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)
datagen.fit(x_train)

# Step 3 Build the model architecture

base_hidden_units = 32
weight_decay = 1e-4
model=Sequential()

#Conv1
model.add(Conv2D(base_hidden_units, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay),
input_shape = x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Conv2
model.add(Conv2D(base_hidden_units, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Pool + Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.2))

#Conv3
model.add(Conv2D(base_hidden_units * 2, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Conv4
model.add(Conv2D(base_hidden_units * 2, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Pool + Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.2))

#Conv5
model.add(Conv2D(base_hidden_units * 4, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Conv6
model.add(Conv2D(base_hidden_units * 4, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# Pool + Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

batch_size = 256
epochs = 10

checkpointer = ModelCheckpoint(filepath=f'model.{epochs}epochs.hdf5', verbose=True, save_best_only=True)
optimizer = keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)

model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), callbacks=[checkpointer],
                              steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, verbose=True,
                              validation_data=(x_valid, y_valid))


# Step 5 evaluate the model
scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('\nTest Result: %.3f loss: %.3f' % (scores[1]*100, scores[0]))

pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
pyplot.savefig(f'model.{epochs}epochs.png')