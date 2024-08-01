# D:\personal projects\sir waleed\Radar_Signal_Characterization\src\load_data.py
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import keras

def load_radchar_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        iq_data = f['iq'][...]  # Load IQ data
        labels_data = f['labels'][...]  # Load labels
    labels = labels_data['signal_type']
    return iq_data, labels

def preprocess_data(file_path):
    X, y = load_radchar_dataset(file_path)
    
    # Split real and imaginary parts into two channels
    X_real = np.real(X)
    X_imag = np.imag(X)
    X = np.stack((X_real, X_imag), axis=-1)

    y = keras.utils.to_categorical(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)
