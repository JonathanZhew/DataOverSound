import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

K = 64 # number of OFDM subcarriers
CP = K//4  # length of the cyclic prefix: 25% of the block
P = 8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits

allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.

# For convenience of channel estimation, let's make the last carrier also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)

plt.figure(figsize=(8,0.8))
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
plt.legend(fontsize=10, ncol=2)
plt.xlim((-1,K)); plt.ylim((-0.1, 0.3))
plt.xlabel('Carrier index')
plt.yticks([])
plt.grid(True)
plt.savefig("pilots.png")

mu = 4 # bits per symbol (i.e. 16QAM)
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}
demapping_table = {v : k for k, v in mapping_table.items()}

channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel
H_exact = np.fft.fft(channelResponse, 2*K)[:K]


SNRdb = 25  # signal to noise-ratio in dB at the receiver 

def SP(bits):
    return bits.reshape((len(dataCarriers), mu))

def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol

def RealizeIDFT(OFDM_data):
    conj_data = np.conjugate(OFDM_data)
    rev_data = conj_data[::-1]
    app_data = np.append([0.0+0.j], rev_data)
    app_data = np.append(app_data, OFDM_data)
    ifft_data = np.fft.ifft(app_data)
    return ifft_data.real

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    
    print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP+2*K+1)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def RealizeDFT(OFDM_RX):
    fft_data = np.fft.fft(OFDM_RX)
    app_data = fft_data[K+1:2*K+1]
    return app_data

def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values

    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase
    # separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)

    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    #plt.ylim(0,2)

    return Hest

def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]

def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])

    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))

    # for each element in QAM, choose the index in constellation
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)

    # get back the real constellation point
    hardDecision = constellation[const_index]

    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

def PS(bits):
    return bits.reshape((-1,))

bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
bits_SP = SP(bits)
QAM = Mapping(bits_SP)
OFDM_data = OFDM_symbol(QAM)

OFDM_time = RealizeIDFT(OFDM_data)
OFDM_withCP = addCP(OFDM_time)
OFDM_TX = OFDM_withCP
OFDM_RX = channel(OFDM_TX)
"""
OFDM_RX = np.array([  932, -1001,   208,    56, -3321, -3366, -2951, -4640, -3915,
        -719,  -502, -3132, -2219, -2151, -1658, -1855,  -781,  1149,
       -4363, -2478, -1446, -9459,  1333, -3588,   814, -1841, -2937,
        4298, -8244,  -451,  3731, -4454,  1702,  4808, -5866, -2041,
       -3030, -4163, -1773, -3538, -6401, -3608, -3095,  -209,   715,
       -2903,  3537, -5826,   205,  4554, -4764,   994, -6133, -4611,
        -215, -6790, -1367, -1880, -1666, -2123, -1100, -1328,  -175,
        1809, -5364,   -29, -3929, -4580,  3818,  -680, -3535,   222,
       -1779,  2220,  1629, -5045, -1040,  3421, -6736, -2582, -2571,
       -1297,  1778,  -807, -5742,  1152,  3086, -6029,  1158, -2514,
        2164,  -823, -2385,  3328,   710,  1713, -5220,  -505,  4785,
       -2111,  5549,  -573, -5925,  2542, -3498,  1820, -1979, -3061,
        3768,   -54, -1728,  4827, -6536,  -736,   758,   -68, -2864,
       -4268,   969, -2255,  2886, -3619, -3450,  1197,  1502,  -635,
        -156,  1544, -4270,  3480,   -34,  2277,  4796, -2901, -1993,
       -1508, -6336, -3577,   355,  -471, -2092, -1615, -2808,  -880,  -418], dtype='int16')
       """
OFDM_RX = np.array([ -1.98004089e-01,   5.16678365e-02,   5.09964293e-02,
         5.23697623e-02,   9.09451582e-02,   3.84594256e-01,
        -2.23395489e-02,   4.43159276e-01,   1.81646168e-01,
         1.06875820e-01,  -7.94701987e-02,   1.13925596e-01,
        -1.37974181e-01,   5.31022065e-02,   2.72988067e-01,
         1.75481429e-02,  -2.00811792e-01,   4.57167272e-02,
         5.00808741e-02,   4.45265053e-02,   7.97143468e-02,
         3.63628040e-01,   1.77648244e-01,  -8.73744926e-02,
         3.60118412e-02,  -1.19968261e-01,  -2.49702445e-01,
         4.98519852e-01,  -1.81463057e-01,   1.15909299e-01,
        -4.38032167e-01,   1.83111057e-04,  -3.24533830e-01,
        -4.92873928e-01,   1.77983947e-01,   6.32953887e-02,
        -1.78319651e-01,   3.52580340e-01,   5.26749474e-01,
        -2.14026307e-01,   1.56682028e-01,   2.73079623e-01,
         3.04208502e-01,  -6.06402783e-02,  -1.81890316e-02,
        -1.91259499e-01,  -1.55339213e-02,  -4.06811731e-02,
         2.27790155e-01,  -5.34073916e-02,  -3.86364330e-02,
         6.28681295e-03,  -2.99111911e-01,   4.67360454e-01,
         2.71584216e-01,  -1.38889737e-01,   7.09860530e-02,
        -1.70751061e-01,   1.63060396e-01,  -4.51368755e-02,
        -1.23294778e-02,   7.40379040e-02,   4.45570238e-03,
         1.57292398e-01,  -7.57164220e-02,   2.18878750e-01,
         1.94708090e-02,   1.93212683e-01,  -7.18710898e-02,
        -2.31574450e-01,   2.08746605e-02,  -2.23487045e-01,
         2.58796960e-02,   1.13223670e-01,  -2.37281411e-01,
        -2.19458602e-01,   1.29490036e-01,   1.60863063e-01,
         2.90108951e-01,   1.46183660e-02,  -1.35624256e-01,
         1.91442610e-01,   2.05938902e-01,  -4.14349803e-01,
         6.73848689e-02,   8.28882717e-02,   2.78359325e-01,
        -1.30161443e-01,  -7.84325694e-02,  -2.85958434e-01,
         3.21298868e-01,  -1.90923795e-01,  -3.95550401e-01,
         1.26956999e-02,  -7.19016083e-02,  -5.13016144e-02,
        -4.00494400e-01,   1.22074038e-02,  -9.38749351e-02,
        -3.41441084e-01,  -3.00729392e-01,  -3.88103885e-01,
         2.48451186e-01,   1.87231056e-01,  -1.30832850e-01,
         5.33768731e-02,  -3.80748924e-01,  -2.18115787e-01,
         3.94604328e-02,  -1.86986908e-01,  -1.21585742e-01,
        -3.76750999e-01,   1.40171514e-01,   2.12591937e-01,
        -3.38755455e-02,   3.69273965e-03,   1.94769127e-01,
        -3.85784478e-01,   3.31827754e-01,   1.41209143e-01,
         2.42622150e-01,  -3.50169378e-01,   1.63029878e-01,
         1.91686758e-01,  -2.68562883e-03,   1.74047060e-01,
        -2.45796075e-01,   5.24002808e-02,   8.99685659e-02,
        -6.35090182e-02,  -3.89538255e-01,  -1.83324686e-01,
        -2.14880825e-01,  -3.69853816e-01,   3.68236335e-01,
        -2.54524369e-02,   4.36902982e-01,   1.67882321e-01,
         1.11362041e-01,  -7.96533097e-02,   1.16458632e-01,
        -1.36173589e-01,   4.95315409e-02,   2.71340068e-01,
         1.78533280e-02,  -1.96905423e-01,   4.83718375e-02,
         4.93179113e-02,   5.57267983e-02])

OFDM_RX2 = OFDM_RX[3:]
OFDM_RX_noCP = removeCP(OFDM_RX2)
OFDM_demod = RealizeDFT(OFDM_RX_noCP)
plt.figure();
Hest = channelEstimate(OFDM_demod)
#plt.savefig("channelEstimate.png")
equalized_Hest = equalize(OFDM_demod, Hest)
QAM_est = get_payload(equalized_Hest)
PS_est, hardDecision = Demapping(QAM_est)
bits_est = PS(PS_est)

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
