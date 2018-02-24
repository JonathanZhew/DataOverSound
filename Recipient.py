import numpy as np
import matplotlib.pyplot as plt
from LibOFDM import *
import soundfile as sf
import sounddevice as sd
import sys
def FindCp(data):
    clen = len(data)-290
    corr_data=[]
    for i in range(clen+1):
        corr = np.corrcoef(data[i:i+32],data[258+i:290+i])
        corr_data.append(float(corr[0,1]))
    arraryCorr = np.array(corr_data)    
    index = np.argmax(arraryCorr)
    print(arraryCorr[index])
    if(arraryCorr[index]<0.85):
        return -1
    else:
        return index

data, fs = sf.read("SenderInt16.wav", dtype='int16')
"""
fs=44100
duration = 1  # seconds
sd.default.samplerate = fs
sd.default.channels = 1
recording = sd.rec(int(duration * fs),dtype='int16')
sd.wait()
print("Recording end!")

data = np.hstack(recording)
sf.write("Recipient.wav", data, fs)
print("save Recording.wav")
"""
#Find head of OFDM Frame    
index = FindCp(data)
print("Frame start at "+ str(index))
if(index == -1):
    print("I can NOT find any singal")
    sys.exit()
OFDM_RX_Int16 = data[index:290+index]

bits_est = ofdm_decode(OFDM_RX_Int16)

bit_dat = np.packbits(bits_est)
print ("".join([chr(item) for item in bit_dat]))

plt.plot(data[index-100:390+index], label= "frame")

plt.figure()
bits_org = np.load('send_bits.npy')
bits_delta = bits_org-bits_est
print ("Obtained Bit error rate: ", np.sum(abs(bits_delta))/len(bits_org))
plt.plot(bits_delta, label='err')

"""
plt.figure()
for qam, hard in zip(QAM_est, hardDecision):
    plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
    plt.plot(hardDecision.real, hardDecision.imag, 'ro')
plt.grid(True);
plt.xlabel('Real part');
plt.ylabel('Imaginary part');
plt.title('Hard Decision demapping');

plt.figure()
#plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
plt.grid(True);
plt.xlabel('Carrier index');
plt.ylabel('$|H(f)|$');
plt.legend(fontsize=10)

plt.show()

plt.figure(figsize=(8,2))
plt.plot(abs(OFDM_TX), label='TX signal')
plt.plot(abs(OFDM_RX), label='RX signal')
plt.legend(fontsize=10)
plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
plt.grid(True);
#plt.savefig("Time-domainSignals.png")

plt.figure()
plt.plot(QAM_est.real, QAM_est.imag, 'bo');
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary Part'); plt.title("Received constellation"); 
#plt.savefig("Constellation.png")

plt.figure()
for qam, hard in zip(QAM_est, hardDecision):
    plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
    plt.plot(hardDecision.real, hardDecision.imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Hard Decision demapping');
#plt.savefig("HardDecision.png")

print ("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))

plt.show()
"""
