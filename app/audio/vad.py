from array import array
import sys
from time import perf_counter


class RMSVAD:
    """Simple PCM gate; Deepgram endpointing remains final turn authority."""

    def __init__(self, threshold: int = 420, end_silence_ms: int = 550, sample_rate: int = 16000, start_voice_ms: int = 160):
        self.threshold, self.end_silence_ms, self.sample_rate = threshold, end_silence_ms, sample_rate
        self.start_voice_ms = start_voice_ms
        self.candidate_voice_ms = 0.0
        self.speaking = False
        self.silence_ms = 0.0
        self.speech_started_at: float | None = None
        self.last_voice_at: float | None = None
        self.ended_at: float | None = None

    def feed(self, pcm: bytes) -> tuple[bool, bool]:
        if not pcm or len(pcm) % 2:
            return False, False
        duration_ms = len(pcm) / 2 / self.sample_rate * 1000
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        level = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
        voiced = level >= (self.threshold * 0.7 if self.speaking else self.threshold)
        started = ended = False
        if voiced:
            if not self.speaking:
                self.candidate_voice_ms += duration_ms
                if self.candidate_voice_ms >= self.start_voice_ms:
                    self.speaking = True
                    self.speech_started_at = perf_counter()
                    started = True
            if self.speaking:
                self.silence_ms = 0
                self.last_voice_at = perf_counter()
        elif self.speaking:
            self.silence_ms += duration_ms
            if self.silence_ms >= self.end_silence_ms:
                self.speaking = False
                self.silence_ms = 0
                self.ended_at = perf_counter()
                ended = True
                self.candidate_voice_ms = 0
        else:
            self.candidate_voice_ms = 0
        return started, ended
