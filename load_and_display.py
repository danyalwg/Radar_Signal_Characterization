import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Mapping of signal_type integers to names
signal_type_mapping = {
    0: 'coherent_pulse_train',
    1: 'barker_code',
    2: 'polyphase_barker_code',
    3: 'frank_code',
    4: 'linear_frequency_modulated'
}

# Load the dataset
def load_radchar_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        iq_data = f['iq'][...]  # IQ data
        labels = f['labels'][...]  # Label data
    return iq_data, labels

# Visualize a given radar waveform
def visualize_waveform(iq_data, labels, idx):
    # Compute time axis
    sps = 3.2e6  # Sampling rate
    n = len(iq_data[idx])
    tmax = n / sps
    t = np.linspace(0, tmax, n)  # Time horizon

    # Clear the previous plot
    ax.cla()
    fig.texts.clear()

    # Plot the in-phase and quadrature components
    ax.plot(t, np.real(iq_data[idx]), marker='.', markersize=4, color='tab:blue', linestyle='-', linewidth=1.5, alpha=1, label='In-phase')
    ax.plot(t, np.imag(iq_data[idx]), marker='None', markersize=4, color='tab:orange', linestyle='-', linewidth=1.5, alpha=0.75, label='Quadrature')

    # Using scientific notation for x-axis
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Metadata annotations
    label = labels[idx]
    annotation = (f"Index: {label[0]}\n"
                  f"Signal Type: {signal_type_mapping[label[1]]}\n"
                  f"Number of Pulses: {label[2]}\n"
                  f"Pulse Width: {label[3]:.2e} s\n"
                  f"Time Delay: {label[4]:.2e} s\n"
                  f"Pulse Repetition Interval: {label[5]:.2e} s\n"
                  f"Signal-to-Noise Ratio: {label[6]} dB")

    # Add the annotations above the plot
    fig.text(0.1, 0.95, annotation, fontsize=10, ha='left', va='top')

    # Set plot labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.set_title(f'Waveform Index: {idx}')
    fig.canvas.draw()

# Event handler for button press
def on_button_click(event):
    global current_idx
    if event.name == 'key_press_event':
        if event.key == 'right':
            current_idx = (current_idx + 1) % len(iq_data)
        elif event.key == 'left':
            current_idx = (current_idx - 1) % len(iq_data)
    elif event.name == 'button_press_event':
        if event.button == 1:  # Left click
            current_idx = (current_idx + 1) % len(iq_data)
        elif event.button == 3:  # Right click
            current_idx = (current_idx - 1) % len(iq_data)
    visualize_waveform(iq_data, labels, current_idx)

# Path to the RadChar-Tiny dataset file
file_path = './RadChar-Tiny.h5'

# Load the dataset
iq_data, labels = load_radchar_dataset(file_path)

# Initial waveform index
current_idx = 0

# Create a figure and visualize the initial waveform
fig, ax = plt.subplots()
visualize_waveform(iq_data, labels, current_idx)

# Connect the key press event handler
fig.canvas.mpl_connect('key_press_event', on_button_click)
fig.canvas.mpl_connect('button_press_event', on_button_click)

# Show the plot
plt.show()
