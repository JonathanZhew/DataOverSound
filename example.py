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
OFDM_RX = np.array([   192,   4932, -15344, -16577,   3504,     20,  -1813,   5072,
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
OFDM_RX = OFDM_RX/0x7FFF

OFDM_RX_noCP = removeCP(OFDM_RX)
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
