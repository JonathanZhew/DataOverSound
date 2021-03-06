import numpy as np
import scipy
from scipy import interpolate
from scipy.interpolate import interp1d

#configure paremeter
K = 64 # number of OFDM subcarriers
CP = K//4  # length of the cyclic prefix: 25% of the block
P = 8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits

allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

paddingCarriers = allCarriers[::K//P] # pad empty for real channel

pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
# For convenience of channel estimation, let's make the last carriers also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)

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
H_exact = np.fft.fft(channelResponse, K)

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
    return Hest, Hest_at_pilots

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

#Over Sample
def OverSample(data, rate = 1/2):
    #y = data
    y = np.hstack([data, np.array([0])])
    x = np.arange(len(y))
    f = interpolate.interp1d(x, y)
    f2 = interpolate.interp1d(x, y, kind='cubic')
    xOverSample = np.arange(0, len(y)-1, rate)
    yOverSample = f(xOverSample)   # use interpolation function returned by `interp1d`   
    return yOverSample

def Sample(data, rate = 1/2):
    K = len(data)
    P = int(K*rate)
    allIndexs= np.arange(K) 
    sampleIndexs= allIndexs[::K//P] 
    ySample = data[sampleIndexs]
    xSample = np.arange(len(ySample))
    return ySample

# bits shoud be length of payloadBits_per_OFDM
def ofdm_encode(bits):
    #bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    bitsLen = len(bits)
    if(bitsLen >= payloadBits_per_OFDM):
        fullBits = bits[:payloadBits_per_OFDM]
    else:
        bPadding = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM - bitsLen, ))
        fullBits = np.hstack([bits, bPadding])
        
    bits_SP = SP(fullBits)
    QAM = Mapping(bits_SP)
    OFDM_data = OFDM_symbol(QAM)

    OFDM_time = RealizeIDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OverSample(OFDM_withCP)

    #float to int16
    symbol = (np.array(OFDM_TX)*0x3FFF).astype(np.int16)

    return symbol

def ofdm_decode(symbol):
    #sampling
    OFDM_RX_Sampled = Sample(symbol)

    #int16 to float
    #OFDM_RX = np.array(OFDM_RX_Sampled)/0x3FFF
    OFDM_RX= OFDM_RX_Sampled

    OFDM_RX_noCP = removeCP(OFDM_RX)
    OFDM_demod = RealizeDFT(OFDM_RX_noCP)
    Hest, Hest_at_pilots = channelEstimate(OFDM_demod)
    #plt.savefig("channelEstimate.png")
    equalized_Hest = equalize(OFDM_demod, Hest)
    QAM_est = get_payload(equalized_Hest)
    PS_est, hardDecision = Demapping(QAM_est)
    bits_est = PS(PS_est)

    return bits_est

