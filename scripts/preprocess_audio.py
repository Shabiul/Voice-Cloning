# import os
# from pydub import AudioSegment
# import noisereduce as nr
# import numpy as np
# import scipy.io.wavfile as wav

# def reduce_noise(input_file, output_file):
#     """
#     Reduce noise in the given audio file and save the result to a new file.
    
#     Args:
#     - input_file (str): Path to the input audio file.
#     - output_file (str): Path to the output audio file where processed audio will be saved.
#     """
#     # Load audio file
#     audio = AudioSegment.from_wav(input_file)
#     audio_np = np.array(audio.get_array_of_samples())
#     rate = audio.frame_rate
    
#     # Reduce noise
#     reduced_noise = nr.reduce_noise(y=audio_np, sr=rate)
    
#     # Save reduced noise audio
#     wav.write(output_file, rate, reduced_noise.astype(np.int16))

# def preprocess_audio(raw_dir, processed_dir):
#     """
#     Preprocess all audio files in the raw recordings directory and save them to the processed audio directory.
    
#     Args:
#     - raw_dir (str): Path to the directory containing raw audio files.
#     - processed_dir (str): Path to the directory where processed audio files will be saved.
#     """
#     # Ensure that the processed directory exists
#     os.makedirs(processed_dir, exist_ok=True)
    
#     # List all files in the raw recordings directory
#     files = [f for f in os.listdir(raw_dir) if f.endswith('.wav')]
    
#     for file_name in files:
#         input_file = os.path.join(raw_dir, file_name)
        
#         # Define the path for the processed audio file
#         output_file = os.path.join(processed_dir, file_name)
        
#         try:
#             # Reduce noise and save the processed file
#             reduce_noise(input_file, output_file)
#             print(f"Processed {file_name} and saved to {output_file}")
#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")

# # Set the directories
# raw_audio_dir = "data/raw_recordings"
# processed_audio_dir = "data/processed_audio"

# # Preprocess the audio files
# preprocess_audio(raw_audio_dir, processed_audio_dir)






# import os
# from pydub import AudioSegment
# import noisereduce as nr
# import numpy as np
# import torch

# def reduce_noise(input_file, output_file):
#     """
#     Reduce noise in the given audio file and save the result to a new file.
    
#     Args:
#     - input_file (str): Path to the input audio file.
#     - output_file (str): Path to the output audio file where processed audio will be saved.
#     """
#     # Load audio file
#     audio = AudioSegment.from_wav(input_file)
#     audio_np = np.array(audio.get_array_of_samples())
#     rate = audio.frame_rate
    
#     # Reduce noise
#     reduced_noise = nr.reduce_noise(y=audio_np, sr=rate)
    
#     # Convert numpy array to PyTorch tensor and reshape it
#     # Assume that you want 1 channel (mono) audio with shape [1, length]
#     tensor = torch.tensor(reduced_noise, dtype=torch.float32).unsqueeze(0)
    
#     # Save tensor as .pt file
#     torch.save(tensor, output_file)

# def preprocess_audio(raw_dir, processed_dir):
#     """
#     Preprocess all audio files in the raw recordings directory and save them to the processed audio directory.
    
#     Args:
#     - raw_dir (str): Path to the directory containing raw audio files.
#     - processed_dir (str): Path to the directory where processed audio files will be saved.
#     """
#     # Ensure that the processed directory exists
#     os.makedirs(processed_dir, exist_ok=True)
    
#     # List all files in the raw recordings directory
#     files = [f for f in os.listdir(raw_dir) if f.endswith('.wav')]
    
#     for file_name in files:
#         input_file = os.path.join(raw_dir, file_name)
        
#         # Define the path for the processed audio file
#         output_file = os.path.join(processed_dir, file_name.replace('.wav', '.pt'))
        
#         try:
#             # Reduce noise and save the processed file as a .pt file
#             reduce_noise(input_file, output_file)
#             print(f"Processed {file_name} and saved to {output_file}")
#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")

# # Set the directories
# raw_audio_dir = "data/raw_recordings"
# processed_audio_dir = "data/processed_audio"

# # Preprocess the audio files
# preprocess_audio(raw_audio_dir, processed_audio_dir)







import os
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display

def reduce_noise(input_file, output_file):
    """
    Reduce noise in the given audio file and save the result to a new file.
    
    Args:
    - input_file (str): Path to the input audio file.
    - output_file (str): Path to the output audio file where processed audio will be saved.
    """
    # Load audio file
    audio = AudioSegment.from_wav(input_file)
    audio_np = np.array(audio.get_array_of_samples())
    rate = audio.frame_rate
    
    # Reduce noise
    reduced_noise = nr.reduce_noise(y=audio_np, sr=rate)
    
    # Convert numpy array to PyTorch tensor
    tensor = torch.tensor(reduced_noise, dtype=torch.float32)
    
    # Save tensor as .pt file
    torch.save(tensor, output_file)

def audio_to_spectrogram(audio_np, sr):
    """
    Convert audio numpy array to a spectrogram.
    
    Args:
    - audio_np (numpy.ndarray): Audio waveform.
    - sr (int): Sampling rate.
    
    Returns:
    - numpy.ndarray: Spectrogram.
    """
    # Convert audio to a spectrogram
    S = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    return S_dB

def preprocess_audio(raw_dir, processed_dir):
    """
    Preprocess all audio files in the raw recordings directory and save them to the processed audio directory.
    
    Args:
    - raw_dir (str): Path to the directory containing raw audio files.
    - processed_dir (str): Path to the directory where processed audio files will be saved.
    """
    # Ensure that the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # List all files in the raw recordings directory
    files = [f for f in os.listdir(raw_dir) if f.endswith('.wav')]
    
    for file_name in files:
        input_file = os.path.join(raw_dir, file_name)
        
        # Define the path for the processed audio file
        output_file = os.path.join(processed_dir, file_name.replace('.wav', '.pt'))
        
        try:
            # Reduce noise
            reduce_noise(input_file, output_file)
            
            # Load the cleaned audio file
            audio_np = torch.load(output_file).numpy()
            rate = 16000  # Sampling rate, update if necessary
            
            # Convert to spectrogram
            spectrogram = audio_to_spectrogram(audio_np, rate)
            
            # Save the spectrogram as a .pt file
            tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            torch.save(tensor, output_file)
            
            print(f"Processed {file_name} and saved to {output_file}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Set the directories
raw_audio_dir = "data/raw_recordings"
processed_audio_dir = "data/processed_audio"

# Preprocess the audio files
preprocess_audio(raw_audio_dir, processed_audio_dir)
