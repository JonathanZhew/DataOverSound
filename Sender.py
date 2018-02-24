import numpy as np
import matplotlib.pyplot as plt
from LibOFDM import *
import soundfile as sf
import sounddevice as sd
import time

fs = 44100

#input data
info = 'Hello! My name is Jonatan.how about you1234561423534647576867857575756767'
a = np.fromstring(info, dtype='uint8')  #print ("".join([chr(item) for item in b]))
b = np.unpackbits(a)                    #b = np.packbits(bits)
#bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
bits = b[:payloadBits_per_OFDM]
print ("Bits count: ", len(bits))
print ("First 20 bits: ", bits[:20])
print ("Mean of bits (should be around 0.5): ", np.mean(bits))

np.save('send_bits', bits)
symbol = ofdm_encode(bits)

#padding zero
pads = np.zeros(100,dtype=np.int16)
OFDM_TX_int16_pad = np.hstack([pads, symbol, pads])
#Save file
sf.write("SenderInt16.wav", OFDM_TX_int16_pad, fs)
print ("SenderInt16.wav Save done!")

#play sound
print("play file len", len(OFDM_TX_int16_pad))
sd.play(OFDM_TX_int16_pad, fs, device=3)
print("play...")
status = sd.wait()

for n in range(10):
    sd.play(OFDM_TX_int16_pad,fs, device=3)
    time.sleep(0.5)
    print("play..."+str(n))
print("The end!")
