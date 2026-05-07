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
CHUNK_SIZE = 48000  # 1-second increments

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


def start_recording():
    recorded_chunks = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(SAVE_FOLDER, f"manual_rec_{timestamp}.wav")

    try:
        # Setup Device
        default_speaker = sc.default_speaker()
        mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)

        print(f"\n--- RECORDING STARTING ---")
        print(f"Device: {default_speaker.name}")
        print("ACTION: Press 'Ctrl + C' to STOP recording and return to menu.")
        print("Recording", end="", flush=True)

        with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
            while True:
                chunk = recorder.record(numframes=CHUNK_SIZE)
                recorded_chunks.append(chunk)
                print(".", end="", flush=True)

    except KeyboardInterrupt:
        # This catches Ctrl+C and proceeds to saving
        print("\n\nStopping... Processing audio...")

        if recorded_chunks:
            full_audio = np.concatenate(recorded_chunks, axis=0)

            print("Trimming starting and ending silence...")
            # Transpose for librosa, trim, then transpose back
            trimmed_data, _ = librosa.effects.trim(
                full_audio.T, top_db=SILENCE_THRESHOLD
            )
            final_data = trimmed_data.T

            sf.write(output_path, final_data, SAMPLE_RATE, subtype="PCM_24")

            print(f"Success! File saved as: {output_path}")
            print(f"Total duration: {len(final_data)/SAMPLE_RATE:.2f} seconds")
        else:
            print("Nothing was recorded.")


def main_menu():
    while True:
        print("\n" + "=" * 30)
        print("      AUDIO RECORDER")
        print("=" * 30)
        print("1. Record Audio")
        print("2. Exit")
        choice = input("\nSelect an option (1-2): ")

        if choice == "1":
            start_recording()
        elif choice == "2":
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        print(f"\nA critical error occurred: {e}")
