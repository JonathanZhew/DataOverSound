import queue
import argparse
import sys
from LibOFDM import *
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import matplotlib.pyplot as plt
def FindCp(data):
    clen = len(data)-290
    corr_data=[]
    for i in range(clen+1):
        corr = np.corrcoef(data[i:i+32],data[258+i:290+i])
        corr_data.append(float(corr[0,1]))
    arraryCorr = np.array(corr_data)    
    index = np.argmax(arraryCorr)
    print(arraryCorr[index])
    if(arraryCorr[index]<0.5):
        return -1
    else:
        return index

def first_index_gt(data_list, value):
    '''return the first index greater than value from a given list like object'''
    try:
        index = next(data[0] for data in enumerate(data_list) if data[1] > value)
        return index
    except StopIteration: return - 1

print(sd.query_devices())

device_info = sd.query_devices(1, 'input')
# soundfile expects an int, sounddevice provides a float:
mysamplerate = int(device_info['default_samplerate'])

#filename = tempfile.mktemp(prefix='rec_unlimited_', suffix='.wav', dir='')
q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

data = np.zeros(10)
symbol = np.array([1])
Count = 0
with sd.InputStream(samplerate=mysamplerate, device=1, channels=1, callback=callback):
    data = q.get();
    while True:
        #time.sleep(5)
        lastdata = data
        data = q.get();
        #stddev = np.std(q.get())
        
        if(Count==0):
            index = first_index_gt(data, 0.01)
            if(index != -1):
                #if(stddev > 0.01):                
                symbol = np.vstack((lastdata, data))
                #plt.plot(symbol)
                #plt.show()
                Count=Count+1
        else:
            Count=Count+1
            #plt.plot(data)
            #plt.show()
            symbol = np.vstack((symbol, data))            
            if(Count==2):
                Count =0
                symbol = np.reshape(symbol,-1)
                #print(len(symbol))                
                #print(symbol)
                #symbol, fs = sf.read("SenderInt16.wav", dtype='int16')
                #index = [ n for n,i in enumerate(symbol) if i>0.01 ][0]
                index = first_index_gt(symbol, 0.01)
                if(index == -1):
                    continue
                
                symbol = symbol[index-10:index+500]
                #plt.plot(symbol)
                #plt.show()
                print(index)
                index = FindCp(symbol)
                print("Frame start at "+ str(index))
                if(index == -1):
                    print("I can NOT find any singal")
                else:
                    OFDM_RX_Int16 = symbol[index:290+index]
                    bits_est = ofdm_decode(OFDM_RX_Int16)
                    bit_dat = np.packbits(bits_est)
                    print ("".join([chr(item) for item in bit_dat]))
                    #print(len(q.get()))

