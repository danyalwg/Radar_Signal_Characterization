# D:\personal projects\sir waleed\Radar_Signal_Characterization\src\train_model.py

import os
import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Add, Activation
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from hyperparameters import EPOCHS, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE
from load_data import preprocess_data

# Load and preprocess data
trainX, testX, trainy, testy = preprocess_data('../data/RadChar-Tiny.h5')
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2])
testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2])

def residual_block(x, filters, kernel_size, activation='relu'):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    
    # Adjust the shortcut to match the number of filters
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    
    x = Add()([shortcut, x])
    x = Activation(activation)(x)
    return x

# Define the ResNet model
def create_resnet_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = residual_block(x, 64, 3)
    x = MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, 128, 3)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

input_shape = (trainX.shape[1], trainX.shape[2])
num_classes = trainy.shape[1]
model = create_resnet_model(input_shape, num_classes)

# Custom callback to print progress
class PrintProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f" - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

# Callbacks
checkpoint = ModelCheckpoint('../models/best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
history = model.fit(trainX, trainy, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[checkpoint, early_stopping, reduce_lr, PrintProgress()], verbose=1)

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

    # Confusion Matrix
    predictions = model.predict(testX)
    predictions = np.argmax(predictions, axis=1)
    true_labels = np.argmax(testy, axis=1)
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('../plots/confusion_matrix.png')

    # Classification Report
    report = classification_report(true_labels, predictions, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])
    print('Classification Report:\n', report)
    with open('../plots/classification_report.txt', 'w') as f:
        f.write(report)

plot_results(history)
