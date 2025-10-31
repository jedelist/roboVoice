import sounddevice as sd
import numpy as np
from queue import Queue

class AudioStream:
    def __init__(self, samplerate=16000, block_ms=30, device_index=None):
        self.sr = samplerate
        self.block = int(self.sr * block_ms / 1000)
        self.q = Queue()
        self.device_index = device_index

    def _callback(self, indata, frames, time, status):
        if status: print(status)
        self.q.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(
            channels=1,
            callback=self._callback,
            samplerate=self.sr,
            blocksize=self.block,
            device=self.device_index,   # set in the confix.yml
            dtype='float32'
        )
        self.stream.start()

    def read(self):
        return self.q.get()

    def stop(self):
        self.stream.stop()
        self.stream.close()
