import sounddevice as sd
import cv2
import numpy as np
import queue
import threading

class StreamHandler:
    """
    Handles the capturing of audio and video streams in synchronized chunks.
    (Final corrected version with audio truncation)
    """
    def __init__(self, duration=5, samplerate=16000, channels=1):
        self.duration = duration
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_size = int(self.duration * self.samplerate)

        self.audio_queue = queue.Queue()
        self.video_capture = cv2.VideoCapture(0)

        if not self.video_capture.isOpened():
            raise IOError("Cannot open webcam")

        self.is_running = False
        self.audio_thread = threading.Thread(target=self._start_audio_stream)

    def _audio_stream_callback(self, indata, frames, time, status):
        """
        This is the "middleman" function.
        It's called by sounddevice and puts the audio data into the queue.
        """
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def _start_audio_stream(self):
        """Function to run the audio stream in a thread."""
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._audio_stream_callback
        ):
            while self.is_running:
                sd.sleep(1000)

    def start(self):
        """Starts the audio and video streams."""
        print("Starting audio and video streams...")
        self.is_running = True
        self.audio_thread.start()

    def read_chunk(self):
        """
        Reads a 5-second chunk of audio and corresponding video frames.
        """
        audio_blocks = []
        collected_samples = 0
        while collected_samples < self.chunk_size:
            block = self.audio_queue.get()
            audio_blocks.append(block)
            collected_samples += len(block)

        # THE FIX: Concatenate all blocks and then truncate to the exact chunk size
        audio_chunk = np.concatenate(audio_blocks)[:self.chunk_size]

        video_frames = []
        num_video_frames = int(self.duration * self.video_capture.get(cv2.CAP_PROP_FPS))
        if num_video_frames == 0: # Fallback if FPS is not readable
            num_video_frames = 30 * self.duration

        for _ in range(num_video_frames):
            ret, frame = self.video_capture.read()
            if ret:
                video_frames.append(frame)

        return audio_chunk, video_frames

    def stop(self):
        """Stops the streams and releases resources."""
        print("Stopping streams...")
        self.is_running = False
        if self.audio_thread.is_alive():
            self.audio_thread.join()
        self.video_capture.release()
        cv2.destroyAllWindows()
