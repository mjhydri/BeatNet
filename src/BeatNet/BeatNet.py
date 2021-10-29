# This is the BeatNet user script. First, it extracts the spectral features and then
# feeds them to one of the pre-trained models to get beat/downbeat activations.
# Therefore, it infers beats and downbeats based on one of the two offline and online inference models.

import os
import torch
import numpy as np
from madmom.features import DBNDownBeatTrackingProcessor
from BeatNet.particle_filtering_cascade import particle_filter_cascade
from BeatNet.log_spect import LOG_SPECT
import librosa
import sys
from BeatNet.model import BDA


class BeatNet:
    def __init__(self, model):
        self.sample_rate = 22050
        log_spec_sample_rate = self.sample_rate
        log_spec_hop_length = int(20 * 0.001 * log_spec_sample_rate)  
        log_spec_win_length = int(64 * 0.001 * log_spec_sample_rate) 
        self.proc = LOG_SPECT(sample_rate=log_spec_sample_rate, win_length=log_spec_win_length,
                             hop_size=log_spec_hop_length, n_bands=[24])
        script_dir = os.path.dirname(__file__)
        #assiging a BeatNet CRNN instance to extract joint beat and downbeat activations
        self.model = BDA(272, 150, 2, 'cpu')   #Beat Downbeat Activation detector
        #loading the pre-trained BeatNet CRNN weigths
        if model == 1:  # GTZAN out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_1_weights.pt')), strict=False)
        elif model == 2:  # Ballroom out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_2_weights.pt')), strict=False)
        elif model == 3:  # Rock_corpus out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_3_weights.pt')), strict=False)
        else:
            raise RuntimeError(f'Failed to open the trained model: {model}')
        self.model.eval()


    def process(self, audio_path, inference_model, plot = False):
        if inference_model == "PF":                 # instantiating a Particle Filter decoder - Is Chosen for online inference
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=50, plot=plot)
        elif inference_model == "DBN":                # instantiating an HMM decoder - Is chosen for offline inference
            self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        else:
            raise RuntimeError('inference_model can be either "PF" or "DBN"')
        preds = self.activation_extractor(audio_path)
        if inference_model == "PF":   # Online _ causal
            data = self.estimator.process(preds)
        elif inference_model == "DBN":    # offline _ none-causal
            data = self.estimator(preds)
        return data

    def activation_extractor(self,audio_path):
        with torch.no_grad():
            if isinstance(audio_path, str):
            	audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
            else:
            	audio = audio_path
            feats = self.proc.process_audio(audio).T
            feats = torch.from_numpy(feats)
            feats = feats.unsqueeze(0)
            preds = self.model(feats)[0]  # extracting the activations by passing the feature through the NN
            preds = self.model.final_pred(preds)
            preds = preds.detach().numpy()
            preds = np.transpose(preds[:2, :])
        return preds




# # test
# instance = BeatNet(1)
# Output = instance.process("C:/datasets/testdata/123.mp3", 'PF', , Plot=True)

