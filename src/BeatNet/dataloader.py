import torch
from torch.utils.data import Dataset
import numpy as np
import os
import librosa  # Ensure librosa is imported
from common import *
from log_spect import LOG_SPECT

class BeatNetDataset(Dataset):
    def __init__(self, audio_dir, target_dir):
        self.audio_dir = audio_dir
        self.target_dir = target_dir

        self.audio_path = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.target_path = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.beats')]

        if len(self.audio_path) != len(self.target_path):
            raise ValueError('Number of audio files and target files do not match')

        
        
        
        self.data_names = self._get_data_list()
        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)

        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, win_length=self.log_spec_win_length,
                             hop_size=self.log_spec_hop_length, n_bands=[24], mode = 'online')

    def __len__(self):
        return len(self.audio_path)
    
    def __getitem__(self, idx):
        data = self._get_data(self.audio_path[idx])
        target = self._get_targets(self.target_path[idx])
        return data, target
    
    def _get_data(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        feats = self.proc.process_audio(audio).T
        feats = torch.from_numpy(feats)
        feats = feats.unsqueeze(0) # Assuming you want a 4D tensor [1, C, H, W]
        return feats
    
    def _get_targets(self, target_path):
        target_list = []
        with open(target_path, 'r') as f:
            for line in f:
                parsed = self._text_label_to_float(line)
                target_list.append(parsed)
        
        # Using the shape of features obtained from the first audio file
        sample_feats = self._get_data(self.audio_path[0])
        beat_vector = np.zeros((sample_feats.shape[1], 3))
        
        beat_times = np.array([x[0] for x in target_list]) * self.sample_rate
        
        for time in beat_times:
            spec_frame = min(int(time / self.log_spec_hop_length), beat_vector.shape[0] - 1)
            for n in range(-2, 3):
                if 0 <= spec_frame + n < beat_vector.shape[0]:
                    beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.5
        
        return torch.tensor(beat_vector)
    
    def _get_data_list(self):
        names = []
        for entry in os.scandir(self.target_dir):
            names.append(os.path.splitext(entry.name)[0])
        return names
    
    def _text_label_to_float(self, text):
        allowed = '1234567890. \t'
        filtered = ''.join([c for c in text if c in allowed])
        if '\t' in filtered:
            t = filtered.rstrip('\n').split('\t')
        else:
            t = filtered.rstrip('\n').split(' ')
        return float(t[0]), float(t[1])

if __name__ == '__main__':
    # Test dataloader
    audio_dir = '/home/nikhil/moji/BeatNet/test/test_data/wav'
    target_dir = '/home/nikhil/moji/BeatNet/test/test_data/beats'

    try:
        dataset = BeatNetDataset(audio_dir, target_dir)
        
        # Fetch the first sample
        sample_data, sample_target = dataset[2]
        
        # Print data and target shapes
        print('Input data shape:', sample_data.shape)
        print('Target shape:', sample_target.shape)
        print('Target:', sample_target)
        print('Data', sample_data)
        
        # Print the length of the dataset
        print('Dataset length:', len(dataset))

    except Exception as e:
        print(f"An error occurred: {e}")
