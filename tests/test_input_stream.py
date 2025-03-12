# import pytest
import sounddevice as sd
import numpy as np


class TestAudioStreams:
    def setUp(self):
        """Set up the necessary attributes for the test."""
        self.sr = 44100  # Sampling rate
        self.channels = 1  # Mono audio
        self.blocksize = 1024  # Number of frames per callback
        self.device_index = None  # Use default device

        # Allocate a buffer to hold one block of audio data
        self.audio_buffer = np.zeros((self.blocksize, self.channels))

    def test_audio_stream_with_delay(self):
        """Test input and output audio streams with delay."""
        
        def audio_input_callback(indata, frames, time, status):
            """Callback for input stream (recording)."""
            if status:
                print(f"Input Stream Error: {status}")
            
            # Directly store the input data in the buffer (no need for append)
            self.audio_buffer[:frames] = indata[:frames]
            print(indata)

        def audio_output_callback(outdata, frames, time, status):
            """Callback for output stream (playback)."""
            if status:
                print(f"Output Stream Error: {status}")
            
            # Play back the data from the buffer with a slight delay
            outdata[:, 0] = self.audio_buffer[:frames, 0]

        # Create an input stream (for recording)
        input_stream = sd.InputStream(
            samplerate=self.sr, channels=self.channels, blocksize=self.blocksize,
            callback=audio_input_callback, device=self.device_index
        )

        # Create an output stream (for playback)
        output_stream = sd.OutputStream(
            samplerate=self.sr, channels=self.channels, blocksize=self.blocksize,
            callback=audio_output_callback, device=self.device_index
        )

        try:
            # Start both streams
            output_stream.start()
            input_stream.start()

            # Record and play audio with a slight delay
            sd.sleep(5000)  # 2 seconds to record and playback the input

        except Exception as e:
            print(f"Audio stream failed with error: {e}")

        finally:
            # Stop the streams
            output_stream.stop()
            input_stream.stop()

if __name__ == "__main__":
    test = TestAudioStreams()
    test.setUp()
    test.test_audio_stream_with_delay()
