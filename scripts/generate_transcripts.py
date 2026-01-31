import os
import speech_recognition as sr

def generate_transcripts(audio_dir, transcript_dir):
    """
    Generate transcripts from audio files and save them as text files.
    
    Args:
    - audio_dir (str): Path to the directory containing audio files.
    - transcript_dir (str): Path to the directory where transcripts will be saved.
    """
    os.makedirs(transcript_dir, exist_ok=True)
    recognizer = sr.Recognizer()
    
    files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]

    for file_name in files:
        audio_path = os.path.join(audio_dir, file_name)
        
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                
                transcript = recognizer.recognize_google(audio_data)
                
                transcript_file = os.path.join(transcript_dir, file_name.replace('.wav', '.txt').replace('.mp3', '.txt'))
                
                with open(transcript_file, 'w') as f:
                    f.write(transcript)
                    
                print(f"Transcript for {file_name} saved to {transcript_file}")
        
        except sr.UnknownValueError:
            print(f"Google Web Speech API could not understand audio in {file_name}")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API for {file_name}: {e}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Generate transcripts
processed_audio_dir = "data/processed_audio"
transcript_dir = "data/transcripts"
generate_transcripts(processed_audio_dir, transcript_dir)
