# D:\personal projects\sir waleed\Radar_Signal_Characterization\src\train_model.py

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from hyperparameters import EPOCHS, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE
from load_data import preprocess_data

# Load and preprocess data
trainX, testX, trainy, testy = preprocess_data('../data/RadChar-Tiny.h5')
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2])
testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2])

# Define the model
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

input_shape = (trainX.shape[1], trainX.shape[2])
num_classes = trainy.shape[1]
model = create_model(input_shape, num_classes)

# Custom callback to print progress
class PrintProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f" - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

# ModelCheckpoint callback
checkpoint = ModelCheckpoint('../models/best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model
history = model.fit(trainX, trainy, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[checkpoint, PrintProgress()], verbose=1)

# Plot results
def plot_results(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('../plots/training_accuracy.png')

    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('../plots/training_loss.png')

plot_results(history)
