
import sounddevice as sd
import soundfile as sf

import time
data, fs = sf.read("wavoutput16b.wav",dtype='int16')
status = sd.wait()
print(status)
for n in range(100):
    sd.play(data, fs, device=3)
    time.sleep(0.5)
