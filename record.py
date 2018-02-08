
import sounddevice as sd
import soundfile as sf


fs=44100
duration = 3.5  # seconds
sd.default.samplerate = fs
sd.default.channels = 1

myrecording = sd.rec(int(duration * fs),dtype='int16')
sd.wait()

sf.write("record16b.wav", myrecording, fs)
