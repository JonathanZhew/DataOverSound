"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import time
from scipy import signal

fs=44100
CHUNK = 1024
CHANNELS = 1
RATE = 44100
SIZE = 4
RECORD_SECONDS = 5

x = np.arange(40960)
sin1 = np.sin((x*x)/16000000)
sin2 = np.sin(x/400+np.pi/8)
sin3 = np.sin(x/80)
sin4 = np.sin(x/10)[:1024]
y = sin1


#plt.plot(sin1,'r')
#plt.plot(sin2)
#plt.plot(sin3,'g')
#plt.plot(y)


fft1 = np.fft.fft(y)
plt.plot(abs(fft1[:102]))

duc=signal.convolve(y, sin4, mode='same')/ sum(sin4)

fft2 = np.fft.fft(duc)
plt.plot(abs(fft2))
#ddc = duc*sin4
#fft3 = np.fft.fft(ddc)
#plt.plot(abs(fft3[0:1000]))


plt.show()
