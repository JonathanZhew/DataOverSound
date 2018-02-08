import numpy as np
import matplotlib.pyplot as plt
from OFDM import *

#input data
info = 'Hello! My name is Jonatan.1234561423534647576867857575756767'
a = np.fromstring(info, dtype='uint8')  #print ("".join([chr(item) for item in b]))
b = np.unpackbits(a)                    #b = np.packbits(bits)
#bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
bits = b[:payloadBits_per_OFDM]
print ("Bits count: ", len(bits))
print ("First 20 bits: ", bits[:20])
print ("Mean of bits (should be around 0.5): ", np.mean(bits))

bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
bits_SP = SP(bits)
QAM = Mapping(bits_SP)
OFDM_data = OFDM_symbol(QAM)

OFDM_time = RealizeIDFT(OFDM_data)
OFDM_withCP = addCP(OFDM_time)
OFDM_TX = OFDM_withCP


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
