import os
import pyaudio
import wave
from datetime import datetime

def record_audio(output_file, duration=10, sample_rate=44100, chunk=1024):
    """
    Record audio from the microphone and save it to a file.
    
    Args:
    - output_file (str): Path to the output file where audio will be saved.
    - duration (int): Duration of the recording in seconds.
    - sample_rate (int): Sample rate of the audio.
    - chunk (int): Number of frames per buffer.
    """
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Open a new stream
    stream = audio.open(format=pyaudio.paInt16, 
                        channels=1, 
                        rate=sample_rate, 
                        input=True, 
                        frames_per_buffer=chunk)
    
    print("Recording...")
    
    # Initialize an empty list to hold audio frames
    frames = []
    
    # Record audio in chunks
    for _ in range(int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Finished recording.")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Terminate PyAudio
    audio.terminate()
    
    # Save the audio file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def get_next_filename(directory, prefix="recording_", extension=".wav"):
    """
    Get the next available filename in the directory by appending a unique index.
    
    Args:
    - directory (str): Directory to check for existing files.
    - prefix (str): Prefix for the filename.
    - extension (str): File extension.
    
    Returns:
    - str: The next available filename.
    """
    i = 0
    while True:
        file_name = f"{prefix}{i}{extension}"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1

# Set the output directory
output_dir = "data/raw_recordings"
os.makedirs(output_dir, exist_ok=True)

# Get the next available filename
output_file = get_next_filename(output_dir)

# Record the audio
record_audio(output_file, duration=10)
