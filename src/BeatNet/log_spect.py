# # Author: Mojtaba Heydari <mheydari@ur.rochester.edu>


# from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
# from madmom.audio.stft import ShortTimeFourierTransformProcessor
# from madmom.audio.spectrogram import (
#     FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
#     SpectrogramDifferenceProcessor)
# from madmom.processors import ParallelProcessor, SequentialProcessor
# from BeatNet.common import *
# import numpy as np



# # feature extractor that extracts magnitude spectrogoram and its differences  

# class LOG_SPECT(FeatureModule):
#     def __init__(self, num_channels=1, sample_rate=22050, win_length=2048, hop_size=512, n_bands=[12], mode='online'):
#         sig = SignalProcessor(num_channels=num_channels, win_length=win_length, sample_rate=sample_rate)
#         self.sample_rate = sample_rate
#         self.hop_length = hop_size
#         self.num_channels = num_channels
#         multi = ParallelProcessor([])
#         frame_sizes = [win_length]  
#         num_bands = n_bands  
#         for frame_size, num_bands in zip(frame_sizes, num_bands):
#             if mode == 'online' or mode == 'offline':
#                 frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size) 
#             else:   # for real-time and streaming modes 
#                 frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size, num_frames=4) 
#             stft = ShortTimeFourierTransformProcessor()  # caching FFT window
#             filt = FilteredSpectrogramProcessor(
#                 num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
#             spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
#             diff = SpectrogramDifferenceProcessor(
#                 diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
#             # process each frame size with spec and diff sequentially
#             multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
#         # stack the features and processes everything sequentially
#         self.pipe = SequentialProcessor((sig, multi, np.hstack))

#     def process_audio(self, audio):
#         feats = self.pipe(audio)
#         return feats.T

# if __name__ == '__main__':
#     # test the feature extraction module and get features for a sample audio file
#     audio_path = '/home/nikhil/moji/BeatNet/src/BeatNet/test_data/808kick120bpm.mp3'
#     feats = LOG_SPECT().process_audio(audio_path)
#     print(feats)

# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, SequentialProcessor
from common import FeatureModule
import numpy as np
import sys
sys.path.append('/home/nikhil/moji/BeatNet/src/BeatNet')


# Feature extractor that extracts magnitude spectrogram and its differences

class LOG_SPECT(FeatureModule):
    def __init__(self, num_channels=1, sample_rate=22050, win_length=2048, hop_size=512, n_bands=[12], mode='online'):
        sig = SignalProcessor(num_channels=num_channels, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.num_channels = num_channels
        multi = ParallelProcessor([])
        frame_sizes = [win_length]  
        num_bands = n_bands  

        for frame_size, num_band in zip(frame_sizes, num_bands):
            if mode in ['online', 'offline']:
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size)
            else:  # for real-time and streaming modes 
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size, num_frames=4)

            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            
            filt = FilteredSpectrogramProcessor(num_bands=num_band, fmin=30, fmax=17000, norm_filters=True)
            
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            
            diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
            

            # Process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))

        # Stack the features and process everything sequentially
        self.pipe = SequentialProcessor((sig, multi, np.hstack))

    def process_audio(self, audio):
        feats = self.pipe(audio)
        return feats.T

if __name__ == '__main__':
    
    # Test the feature extraction module and get features for a sample audio file
    audio_path = '/home/nikhil/moji/BeatNet/test/test_data/wav/Albums-AnaBelen_Veneo-01.wav'
    op = LOG_SPECT().process_audio(audio_path)
    print(op.shape)
