import os
import torchaudio
import torch

# Define the folder path and target length (10 seconds)
folder_path = "demo_magnet_s_wav/"
target_length = 10 * 16000  # Assuming a sample rate of 16 kHz

# Function to pad or truncate the waveform to the target length
def pad_waveform(waveform, target_length):
    current_length = waveform.shape[1]
    if current_length < target_length:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :target_length]
    return waveform

# Iterate through the .wav files in the folder and process each
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Pad or truncate to 10 seconds
        padded_waveform = pad_waveform(waveform, target_length)
        
        # Do something with the padded waveform, like saving or further processing
        # Example: Save the padded waveform back to a file
        output_path = os.path.join(folder_path, f"{filename}")
        torchaudio.save(output_path, padded_waveform, sample_rate)
