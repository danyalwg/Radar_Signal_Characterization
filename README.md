# Radar Signal Characterization

This project implements an AI-based radar signal characterization using the RadChar dataset. The project includes data loading, preprocessing, model training, and evaluation. It uses TensorFlow and Keras for building and training the neural network models.

## Project Structure
- `data/`: Contains the RadChar dataset.
  - `RadChar-Tiny.h5`: The dataset file containing radar signal data and labels.
- `models/`: Directory for saving trained models.
- `plots/`: Directory for saving generated plots.
- `src/`: Source code directory.
  - `hyperparameters.py`: Defines the hyperparameters for training.
  - `load_data.py`: Functions for loading and preprocessing the dataset.
  - `train_model.py`: Script for training the model.
- `requirements.txt`: Lists the dependencies required to run the project.

## Directory Structure
Below is the tree structure of the project directory:

```
Radar_Signal_Characterization/
├── data/
│   └── RadChar-Tiny.h5
├── models/
├── plots/
├── src/
│   ├── hyperparameters.py
│   ├── load_data.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

## Requirements
To replicate this project, you need the following dependencies:
- h5py
- numpy
- matplotlib
- tensorflow
- keras
- scikit-learn
- seaborn

You can install all the dependencies using the following command:
```sh
pip3 install -r requirements.txt
```

## How to Run

### Step 1: Clone the Repository
First, clone the repository from GitHub:
```sh
git clone https://github.com/danyalwg/Radar_Signal_Characterization.git
cd Radar_Signal_Characterization
```

### Step 2: Download the Dataset
Download the RadChar dataset from [here](https://github.com/abcxyzi/RadChar.git) and place it in the `data` directory.

### Step 3: Install Dependencies
Install the required packages:
```sh
pip3 install -r requirements.txt
```

### Step 4: Run the Training Script
Run the training script:
```sh
python3 src/train_model.py
```
Note: Ensure that the dataset path in the `train_model.py` file is correctly set to the actual path where the dataset is located.

## Dataset
The RadChar dataset is a synthetic radar signal dataset designed to facilitate the development of multi-task learning models. It contains pulsed radar signals at varying signal-to-noise ratios (SNRs) between -20 to 20 dB. Each dataset comprises a total of 5 radar signal types each covering 4 unique signal parameters. The sampling rate used in RadChar is 3.2 MHz. Each waveform in the dataset contains 512 complex, baseband IQ samples.

### Signal Types
- Barker codes, up to a code length of 13
- Polyphase Barker codes, up to a code length of 13
- Frank codes, up to a code length of 16
- Linear frequency-modulated (LFM) pulses
- Coherent unmodulated pulse trains

### Signal Parameters
- Number of pulses, sampled between uniform range 2 to 6
- Pulse width, sampled between uniform range 10 to 16 µs
- Pulse repetition interval (PRI), sampled between uniform range 17 to 23 µs
- Pulse time delay, sampled between uniform range 1 to 10 µs

## Source Code Details

### hyperparameters.py
This file defines the hyperparameters for training the model. It sets parameters like the number of epochs, batch size, learning rate, and dropout rate.

### load_data.py
This file contains functions to load and preprocess the RadChar dataset.
1. **Loading Data**: Reads the IQ data and labels from the dataset file.
2. **Preprocessing Data**: Splits the real and imaginary parts of the IQ data into two channels, then splits the data into training and validation sets.

### train_model.py
This file handles the training of the machine learning model.
1. **Check GPU Availability**: Checks if a GPU is available and uses it if possible.
2. **Load and Preprocess Data**: Uses functions from `load_data.py` to load and preprocess the dataset.
3. **Model Architecture**: Defines a Convolutional Neural Network (CNN) with residual blocks and an attention layer.
4. **Training the Model**: Compiles and trains the model using the preprocessed data. It includes callbacks for early stopping, learning rate reduction, and model checkpointing.
5. **Evaluation and Plotting**: Generates plots for training/validation accuracy and loss, confusion matrix, precision-recall curve, and ROC curve.

## Model Architecture
The model is built using a combination of Convolutional layers, residual blocks, and an attention layer. It uses the Adam optimizer and categorical cross-entropy loss for training. The model is designed to classify radar signals into different types.

## Results
The following plots and metrics are generated during training and saved in the `plots/` directory:
- Training and Validation Accuracy
- Training and Validation Loss
- Confusion Matrix
- Classification Report
- Precision-Recall Curve
- ROC Curve

These visualizations help in understanding the model's performance and diagnosing any issues during training.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
