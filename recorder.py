import soundcard as sc
import soundfile as sf
import os
import librosa
import numpy as np
from datetime import datetime

# --- Configuration ---
SAMPLE_RATE = 48000 
SAVE_FOLDER = "Recordings"
SILENCE_THRESHOLD = 30 
CHUNK_SIZE = 48000  # Records in 1-second increments

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(SAVE_FOLDER, f"manual_rec_{timestamp}.wav")

try:
    # 1. Setup Device
    default_speaker = sc.default_speaker()
    mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)

    print(f"--- RECORDING START ---")
    print(f"Device: {default_speaker.name}")
    print("ACTION: Press 'Ctrl + C' to STOP recording and save.")
    print("-----------------------")

    recorded_chunks = []

    # 2. Continuous Recording Loop
    with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
        while True:
            # Capture 1 second at a time
            chunk = recorder.record(numframes=CHUNK_SIZE)
            recorded_chunks.append(chunk)
            # Subtle visual cue that it's working
            print(".", end="", flush=True) 

except KeyboardInterrupt:
    # 3. Handle Stop Signal (Ctrl+C)
    print("\n\nStopping... Processing audio...")
    
    if recorded_chunks:
        # Combine all recorded seconds into one big array
        full_audio = np.concatenate(recorded_chunks, axis=0)

        # 4. Trim Silence
        print("Trimming starting and ending silence...")
        # Transpose for librosa, trim, then transpose back
        trimmed_data, _ = librosa.effects.trim(full_audio.T, top_db=SILENCE_THRESHOLD)
        final_data = trimmed_data.T

        # 5. Save High-Quality Output
        sf.write(output_path, final_data, SAMPLE_RATE, subtype='PCM_24')

        print(f"Success! File saved as: {output_path}")
        print(f"Total duration: {len(final_data)/SAMPLE_RATE:.2f} seconds")
    else:
        print("Nothing was recorded.")

except Exception as e:
    print(f"\nAn error occurred: {e}")