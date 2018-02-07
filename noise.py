"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import time

fs=44100
CHUNK = 1024
CHANNELS = 1
RATE = 44100
SIZE = 4
RECORD_SECONDS = 5

noise = np.random.rand(44096)*2-1
sf.write("noise.wav", noise, fs)
org_demo = np.fft.fft(noise[:2048])
plt.plot(abs(org_demo[1:1024]),'-x')
"""
plt.plot(org_data)
plt.plot(data)
plt.show()

print("read file len", len(noise))
sd.play(noise, RATE, device=3)
print("play...")
status = sd.wait()
print("The end!")
for n in range(5):
    sd.play(noise, RATE, device=3)
    time.sleep(0.5)

data, fs = sf.read("Phonerecord.wav",dtype='float32')
sd.play(data, RATE, device=3)
print("play...")
status = sd.wait()
print("The end!")
data = data[10000:12048]
#indices = [i for i,v in enumerate(data >= -0.1) if v]
org_demo = np.fft.fft(data)
plt.plot(abs(org_demo[100:1024])*30,'-o')
plt.show()
"""
