import pyaudio
import numpy as np


class AudioModule():
    def __init__(self, rate, chunk):
        self.chunk = chunk
        self.rate = rate
        self.p = pyaudio.PyAudio()

    def get_audio(self):
        stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.rate, input=True, output=True, frames_per_buffer=self.chunk)
        print("* Recording")
        while True:
            data = np.frombuffer(stream.read(self.chunk), dtype=np.float32)
            yield data