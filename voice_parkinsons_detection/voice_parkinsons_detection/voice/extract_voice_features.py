import parselmouth
import librosa
import numpy as np
import nolds
import pandas as pd

def extract_voice_features(filepath, sr=44100):
    try:
        y, _ = librosa.load(filepath, sr=sr)
        snd = parselmouth.Sound(filepath)
        min_pitch = 75  # Must be > 0

        print(f"Audio duration: {snd.duration}s, samples: {len(y)}")
        print(f"Min pitch set to: {min_pitch}")

        pitch = snd.to_pitch(time_step=0.01, pitch_floor=min_pitch)
        freqs = pitch.selected_array['frequency']
        print("Pitch frequencies sample:", freqs[:10])

        if freqs.size == 0 or np.all(freqs == 0):
            raise ValueError("No pitch detected. Audio may be silent or unclear.")

        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 0, snd.duration, min_pitch)
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, min_pitch, 0.1, 1.0)

        features = {}

        features["MDVP:Fo(Hz)"] = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        features["MDVP:Fhi(Hz)"] = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz")
        features["MDVP:Flo(Hz)"] = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz")

        features["MDVP:Jitter(%)"] = 100 * parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features["MDVP:Jitter(Abs)"] = parselmouth.praat.call(point_process, "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
        features["MDVP:RAP"] = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features["MDVP:PPQ"] = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        features["Jitter:DDP"] = 3 * features["MDVP:RAP"]

        features["MDVP:Shimmer"] = parselmouth.praat.call(snd, "Get shimmer (local)", point_process, 0, 0, 0.0001, 0.02)
        features["MDVP:Shimmer(dB)"] = parselmouth.praat.call(snd, "Get shimmer (local_dB)", point_process, 0, 0, 0.0001, 0.02)
        features["Shimmer:APQ3"] = parselmouth.praat.call(snd, "Get shimmer (apq3)", point_process, 0, 0, 0.0001, 0.02)
        features["Shimmer:APQ5"] = parselmouth.praat.call(snd, "Get shimmer (apq5)", point_process, 0, 0, 0.0001, 0.02)
        features["MDVP:APQ"] = parselmouth.praat.call(snd, "Get shimmer (apq11)", point_process, 0, 0, 0.0001, 0.02)
        features["Shimmer:DDA"] = 3 * features["Shimmer:APQ3"]

        features["HNR"] = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        features["NHR"] = 1 / (features["HNR"] + 1e-5)

        f0, _, _ = librosa.pyin(y, fmin=min_pitch, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]

        features["RPDE"] = np.std(np.log(f0 + 1e-5)) if len(f0) > 0 else 0
        features["DFA"] = nolds.dfa(y)
        features["spread1"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features["spread2"] = np.std(mfcc)
        features["D2"] = nolds.corr_dim(y, emb_dim=10)
        features["PPE"] = np.std(np.diff(np.log(f0 + 1e-5))) if len(f0) > 1 else 0

        return pd.DataFrame([features])

    except Exception as e:
        raise RuntimeError(f"Parselmouth failed to process audio: {e}")
