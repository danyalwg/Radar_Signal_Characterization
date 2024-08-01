import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Add, Activation, Attention
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import seaborn as sns
import numpy as np
from hyperparameters import EPOCHS, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE
from load_data import load_radchar_dataset, preprocess_data
import os

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
for device in physical_devices:
    print(device)

# Use GPU if available
device_name = "/GPU:0" if len(physical_devices) > 0 else "/CPU:0"
print(f"Using device: {device_name}")

# Load and preprocess data
file_path = 'D:/personal projects/sir waleed/Radar_Signal_Characterization/data/RadChar-Tiny.h5'
trainX, valX, trainy, valy = preprocess_data(file_path)

trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2])
valX = valX.reshape(valX.shape[0], valX.shape[1], valX.shape[2])

def residual_block(x, filters, kernel_size, activation='relu'):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    
    x = Add()([shortcut, x])
    x = Activation(activation)(x)
    return x

def create_advanced_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = residual_block(x, 64, 3)
    x = MaxPooling1D(pool_size=2)(x)
    x = residual_block(x, 128, 3)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Attention Layer
    attention = Attention()([x, x])
    x = Add()([x, attention])
    
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(DROPOUT_RATE)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load last checkpoint if exists
checkpoint_path = '../models/last_checkpoint.keras'
with tf.device(device_name):
    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)
    else:
        input_shape = (trainX.shape[1], trainX.shape[2])
        num_classes = trainy.shape[1]
        model = create_advanced_model(input_shape, num_classes)

    # Custom callback to print progress
    class PrintProgress(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print(f" - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

    # Callbacks
    checkpoint_best = ModelCheckpoint('../models/best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    checkpoint_last = ModelCheckpoint('../models/last_checkpoint.keras', monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # Train the model
    history = model.fit(trainX, trainy, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(valX, valy), callbacks=[checkpoint_best, checkpoint_last, early_stopping, reduce_lr, PrintProgress()], verbose=1)

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
        predictions = model.predict(valX)
        predictions = np.argmax(predictions, axis=1)
        true_labels = np.argmax(valy, axis=1)
        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('../plots/confusion_matrix.png')

        # Classification Report
        target_names = ['Coherent Pulse Train', 'Barker Code', 'Polyphase Barker Code', 'Frank Code', 'Linear Frequency Modulated']
        report = classification_report(true_labels, predictions, target_names=target_names)
        print('Classification Report:\n', report)
        with open('../plots/classification_report.txt', 'w') as f:
            f.write(report)

    plot_results(history)

    def plot_precision_recall_curve(valX, valy):
        predictions = model.predict(valX)
        true_labels = np.argmax(valy, axis=1)
        precision = {}
        recall = {}
        for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(valy[:, i], predictions[:, i])

        plt.figure()
        for i in range(num_classes):
            plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.savefig('../plots/precision_recall_curve.png')
        plt.show()

    def plot_roc_curve(valX, valy):
        predictions = model.predict(valX)
        true_labels = np.argmax(valy, axis=1)
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(valy[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('../plots/roc_curve.png')
        plt.show()

    def plot_learning_rate_schedule(history):
        lr = history.history['lr'] if 'lr' in history.history else [LEARNING_RATE] * len(history.history['loss'])
        plt.figure()
        plt.plot(lr)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.savefig('../plots/learning_rate_schedule.png')
        plt.show()

    def plot_combined_accuracy_loss(history):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('../plots/combined_accuracy_loss.png')
        plt.show()

    plot_precision_recall_curve(valX, valy)
    plot_roc_curve(valX, valy)
    plot_learning_rate_schedule(history)
    plot_combined_accuracy_loss(history)
