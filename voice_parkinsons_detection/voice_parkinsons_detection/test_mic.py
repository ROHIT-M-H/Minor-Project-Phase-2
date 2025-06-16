import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

duration = 3  # seconds
fs = 44100
print("🎤 Speak now...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("✅ Done recording")

print("📈 Max amplitude recorded:", np.max(np.abs(recording)))

if np.max(np.abs(recording)) < 0.01:
    print("⚠️ Mic may be silent. Try speaking louder or check mic input.")
else:
    print("✅ Audio detected, mic is working!")

write("test.wav", fs, np.int16(recording * 32767))
