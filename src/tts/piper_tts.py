import threading
import queue
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

class PiperSpeakerThread(threading.Thread):
    def __init__(self, model_path: str, text_queue: queue.Queue):
        super().__init__()
        self.voice = PiperVoice.load(model_path)
        self.queue = text_queue
        self._stop_event = threading.Event()
        speedup = 1.0 # 1.0 = normal, 1.5 = 50% faster, etc.
        self.stream = sd.OutputStream(
            samplerate=int(self.voice.config.sample_rate * speedup),
            channels=1,
            dtype='int16'
        )

    def run(self):
        self.stream.start()
        while not self._stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.1)
                self._speak(text)
                self.queue.task_done()
            except queue.Empty:
                continue
        self.stream.stop()
        self.stream.close()

    def _speak(self, text: str):
        for chunk in self.voice.synthesize(text):
            int_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            self.stream.write(int_data)

    def stop(self):
        self._stop_event.set()