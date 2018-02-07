import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
rate, data = wav.read('noise.wav')
rate2, data2 = wav.read('record16bnoise.wav')
"""
fft_out = fft(data)

plt.plot(data, np.abs(fft_out))
plt.plot(np.abs(fft_out))
plt.figure()

#
plt.figure()
plt.plot(data)


plt.show()
"""
