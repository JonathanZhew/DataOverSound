"""PyAudio Example: Play a WAVE file."""

import pyaudio
import wave
import sys
import struct


CHUNK = 1024

wf = wave.open("output.wav", 'rb')

p = pyaudio.PyAudio()                            
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

data = wf.readframes(CHUNK)
print(len(data))
for i in range(int(len(data)/4)):
    value = struct.unpack('f', data[i*4:i*4+4])
    print(value)
while data != b'':
    #stream.write(data)
    data = wf.readframes(CHUNK)
    print(len(data))


stream.stop_stream()
stream.close()

p.terminate()
