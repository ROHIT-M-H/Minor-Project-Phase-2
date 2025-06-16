import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from voice.extract_voice_features import extract_voice_features
import joblib
import os

def record_audio(filename="voice.wav", duration=3, fs=44100):
    print("🔴 Recording for 3 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    if np.max(np.abs(recording)) < 0.01:
        raise ValueError("Recording appears to be silent. Please speak louder or check your microphone.")

    recording_int16 = np.int16(recording * 32767)
    write(filename, fs, recording_int16)
    print(f"🎙️ Saved recording to {filename}")
    return filename

def main():
    try:
        wav_file = record_audio()
        features_df = extract_voice_features(wav_file, sr=44100)

        model_path = "models/voice_model.pkl"
        if not os.path.exists(model_path):
            print("❌ Model file not found. Train the model first.")
            return

        pipeline = joblib.load(model_path)
        prediction = pipeline.predict(features_df)[0]
        confidence = pipeline.predict_proba(features_df)[0][1] if hasattr(pipeline, "predict_proba") else None

        result = "🧠 Parkinson's" if prediction == 1 else "✅ Healthy"
        if confidence is not None:
            print(f"\nPrediction: {result} (Confidence: {confidence:.2f})")
        else:
            print(f"\nPrediction: {result}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
