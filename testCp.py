import numpy as np
import matplotlib.pyplot as plt
"""PyAudio Example: Play a WAVE file."""

import pyaudio
import wave
import sys
import struct


CHUNK = 1024

wf = wave.open("output5s.wav", 'rb')

data = wf.readframes(1024*100)

float_data=[]
for i in range(int(len(data)/4)):
    value = struct.unpack('f', data[i*4:i*4+4])[0]
    float_data.append(value)

    
corr = np.corrcoef(float_data[0:16],float_data[129:129+16])
print(corr)

pyd = np.array(float_data)
#plt.plot(pyd)


corr_data=[]
for i in range(len(float_data)-129-16):
    corr = np.corrcoef(float_data[i:i+16],float_data[129+i:129+16+i])
    corr_data.append(float(corr[0,1]))
    #print(i)
    #print(corr[0,1])
npCorr = np.array(corr_data[10000:11000])
plt.plot(npCorr)
plt.show() 





