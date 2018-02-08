import numpy as np
import matplotlib.pyplot as plt
"""PyAudio Example: Play a WAVE file."""

import sounddevice as sd
import soundfile as sf

data0, fs0 = sf.read("org_float.wav",dtype='float')
data, fs = sf.read("record16bCrop.wav",dtype='int16')
"""
ODM_RX = np.array([   192,   4932, -15344, -16577,   3504,     20,  -1813,   5072,
       -14758, -16793,   2628,   3806, -15427, -11222, -13847,  -9965,
       -18669,   3061,   3838,  -8151,  24223,  -5597,    224,   4127,
        -7826,  19818,  11174,  15304,   5315, -21639,   3378,   6577,
       -21570,   4311,   4409, -15730, -13868,  -2266,  16580,  14132,
        10545,  18890,   -276,  -7725,  23781,  -4159,   -905,   9537,
       -24136,   6715,  -1399,   -844,   4655,  -8521,  22274,    208,
       -13561, -15597,   1455,   4986, -16233, -13104,  -3662,  21436,
        -3293,  -2259,   8131, -19901,  -1711,  19149,   1454, -16008,
       -10070, -17639,   1539,   7507, -21504,   3773,   4409, -15576,
       -14799,   -954,  14847,  17571,   -946,  -5685,  22112,  -2597,
        -3068,  16069,  16378,    322,  -5434,  18806,  11307,  15105,
         5666, -21870,   5433,   1960,  -7974,  23711,  -3507,  -3303,
        17155,  12550,   5502, -21215,   2352,   7703, -22583,   4709,
         3046, -14099, -15982,   -464,   8468, -22933,   5441,   1267,
        -3530,   8338, -20790,     47,  16118,  11782,  12664,  17578,
        -2088,  -1466,   5193, -15201, -16528,   3920,    262,  -1416,
         4994, -14990, -15842,   1595,   4382, -16010, -12256, -10555,
       -17538,   2163,   3746, -14534, -16769,   3436,    175,  -1901,
         5160, -14752, -16643,   2763,   3818, -15197, -11032, -13718,
        -9763, -18506,   3278,   4110,  -8155,  24349,  -5554,      1], dtype='int16')
plt.plot(data)
plt.plot(ODM_RX/0x7FF)
plt.show()
"""
clen = len(data)-300
corr_data=[]
for i in range(clen):
    corr = np.corrcoef(data[i:i+32],data[258+i:290+i])
    corr_data.append(float(corr[0,1]))

index = np.argmax(corr_data)    
record = -data[index:290+index]
#int16 to float
float_dat = np.array(record)/32767

#sampling
sampleRate = 1/2
K = len(float_dat)
P = int(K*sampleRate)
allIndexs= np.arange(K) 
sampleIndexs= allIndexs[::K//P] 
ySample = float_dat[sampleIndexs]
xSample = np.arange(len(ySample))


plt.plot(data0[:300])
plt.plot(ySample)
plt.show()
"""
corr = np.corrcoef(data[0:32],data[258:290])
print(corr)

pyd = np.array(data)
#plt.plot(pyd)



for i in range(len(data)-129-16):
    corr = np.corrcoef(data[i:i+16],data[129+i:129+16+i])
    corr_data.append(float(corr[0,1]))
    #print(i)
    #print(corr[0,1])

indices = [i for i,v in enumerate(corr_data) if v >= 0.95]

clen = len(data)-333
for i in range(clen):
    corr = np.corrcoef(data[i:i+32],data[258+i:290+i])
    corr_data.append(float(corr[0,1]))
    #print(i)
    #print(corr[0,1])

npCorr = np.array(corr_data)
plt.plot(npCorr)
plt.show() 
"""




