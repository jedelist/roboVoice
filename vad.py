import torch, numpy as np, time

class VADSegmenter:
    def __init__(self, sr=16000, th=0.5, min_speech_ms=200, max_speech_ms=3000, min_silence_ms=300):
        self.sr = sr
        self.th = th
        self.min_speech = int(min_speech_ms/1000*self.sr)
        self.max_speech = int(max_speech_ms/1000*self.sr)
        self.min_silence = int(min_silence_ms/1000*self.sr)
        # load silero vad
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
        (self.get_speech_ts, _, self.read_audio, _, _) = utils

        self.buf = np.zeros(0, dtype=np.float32)
        self.last_speech_ts = None

    def push(self, chunk: np.ndarray):
        # chunk shape: (N, 1)
        mono = chunk.astype(np.float32).flatten()
        self.buf = np.concatenate([self.buf, mono])

        # only evaluate when buffer has decent sized
        if len(self.buf) < self.min_speech: 
            return None

        # use get_speech_ts on current buffer
        ts = self.get_speech_ts(self.buf, self.model, sampling_rate=self.sr, threshold=self.th, min_speech_duration_ms= self.min_speech / self.sr * 1000, min_silence_duration_ms= self.min_silence / self.sr * 1000, return_seconds=False)

        if not ts:  # no speech yet keep buffering but avoid infinite growth
            if len(self.buf) > self.max_speech:
                self.buf = self.buf[-self.max_speech:]
            return None

        # Take the last complete segment if silence comes after here
        last = ts[-1]
        if last.get('end', None) is not None and (len(self.buf) - last['end']) >= self.min_silence:
            start, end = last['start'], last['end']
            segment = self.buf[start:end].copy()
            # drop everything up to end from buffer
            self.buf = self.buf[end:]
            return segment

        # Otherwise keep waiting for silence
        if len(self.buf) > self.max_speech:
            # force cut if too long
            segment = self.buf[:self.max_speech].copy()
            self.buf = self.buf[self.max_speech:]
            return segment
        return None
