# killall -9 watch mpg123
# watch -n0 nohup command &

import os
from scipy.io import wavfile as wf
import threading
import time
import pyaudio
import wave

filename = '/home/apopa/Desktop/...'
url = 'http://edge76.rdsnet.ro:84/digifm/digifm.mp3'
frames = str(170)
rate = str(16000)
command = 'mpg123 --float --quiet --mono --frames ' + frames + ' --rate ' + rate + ' --skip 2 --wav ' + filename + ' --list ' + url
command_out = 'mpg123 --float --quiet --mono --frames ' + frames + ' --rate ' + rate + ' --skip 2 --list ' + url

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "/home/apopa/Desktop/..."
 
audio = pyaudio.PyAudio()
 
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)


frames = []
#3164
def run4():
    global stream, audio, frames
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
audio.open()

def run1():
    global command
    os.system(command)
    
def run2():
    global command_out
    os.system(command_out)
    
t1 = threading.Thread(target=run1)
t2 = threading.Thread(target=run2)
t3 = threading.Thread(target=run4)
t3.start()
t1.start()
t2.start()
t1.join()
t2.join()
t3.join()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
