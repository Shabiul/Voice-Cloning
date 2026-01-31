# Voice Cloning Project

This project provides a comprehensive toolkit for voice cloning, enabling users to record audio, preprocess data, train custom voice models, and synthesize speech. It includes implementations for custom CNN-based models and integrates with popular voice conversion frameworks like RVC (Retrieval-based Voice Conversion) and So-VITS-SVC.

## Features

-   **Audio Recording**: Capture high-quality audio samples directly from your microphone.
-   **Audio Preprocessing**: Tools for noise reduction and data formatting to prepare audio for training.
-   **Model Training**: Train custom Voice Cloning models (CNN-based) on your dataset.
-   **Voice Synthesis**: Generate speech from text using trained models.
-   **Advanced Frameworks**: Includes resources for RVC and So-VITS-SVC workflows.

## Directory Structure

-   `scripts/`: Python scripts for core functionality.
    -   `record_audio.py`: Script to record audio samples.
    -   `preprocess_audio.py`: Utilities for cleaning and converting audio files.
    -   `train_model.py`: Training loop and model architecture (`SimpleCNN`).
    -   `synthesize_voice.py`: Inference script to generate audio from text.
-   `RVC/`: Resources and notebooks for RVC model training and inference.
-   `so-vits-svc/`: Resources for So-VITS-SVC.
-   `Dataset/` & `AudioData/`: Storage for raw and processed audio datasets.
-   `sileroTTS.ipynb`: Notebook for Silero TTS experiments.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Shabiul/Voice-Cloning.git
    cd Voice-Cloning
    ```

2.  Install the required dependencies:
    ```bash
    pip install torch torchaudio numpy pydub noisereduce scipy pyaudio
    ```
    *(Note: You may need to install PyTorch with CUDA support if you plan to train on GPU.)*

## Usage

### 1. Recording Audio
Record your own voice samples to create a dataset:
```bash
python scripts/record_audio.py
```
This will save recorded `.wav` files to `data/raw_recordings`.

### 2. Preprocessing
Clean your recorded audio (noise reduction) and prepare it for training:
```bash
python scripts/preprocess_audio.py
```

### 3. Training the Model
Train the custom voice cloning model:
```bash
python scripts/train_model.py
```
Adjust parameters like `epochs` and `batch_size` inside the script as needed.

### 4. Synthesizing Voice
Generate audio using your trained model:
```bash
python scripts/synthesize_voice.py
```

## Git LFS
This repository uses Git LFS (Large File Storage) for handling large audio and model files. Ensure you have Git LFS installed:
```bash
git lfs install
git lfs pull
```

## License
[MIT License](LICENSE)
