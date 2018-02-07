import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

import time

org_data, fs = sf.read("wavoutput16b.wav",dtype='int16')

data, fs = sf.read("recordonline2.wav",dtype='int16')

"""
plt.plot(org_data)
plt.plot(data)
plt.show()
"""
print("read file len", len(data))
sd.play(data, fs, device=3)
print("play...")
status = sd.wait()



for n in range(5):
    sd.play(data,fs, device=3)
    time.sleep(0.5)
print("The end!")

corr_data=[]
"""
for i in range(len(data)-129-16):
    corr = np.corrcoef(data[i:i+16],data[129+i:129+16+i])
    corr_data.append(float(corr[0,1]))
    #print(i)
    #print(corr[0,1])
"""
for i in range(len(data)-580):
    corr = np.corrcoef(data[i:i+64],data[516+i:580+i])
    corr_data.append(float(corr[0,1]))
npCorr = np.array(corr_data)
plt.plot(npCorr)
plt.plot(data/0xFFFF)
plt.show()

indices = [i for i,v in enumerate(npCorr >= 0.8) if v]

frame = data[1459:1459+580]
plt.plot(frame,'r-x')
plt.plot(org_data[:580],'g-o')
plt.show()

fft1 = np.fft.fft(frame)
fft2 = np.fft.fft(org_data[:580])
plt.plot(abs(fft1[:128]),'r-x')
plt.plot(abs(fft2[:128]),'g-o')
plt.show()
#sampling
sampleRate = 1/4
K = len(frame)
P = int(K*sampleRate)
allIndexs= np.arange(K) 
sampleIndexs= allIndexs[::K//P] 
ysam = frame[sampleIndexs]
xsam = np.arange(len(frame))

