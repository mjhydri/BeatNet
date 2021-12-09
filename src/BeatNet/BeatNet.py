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
import pyaudio
import matplotlib.pyplot as plt
import time
import threading


class BeatNet:

    '''
    The main BeatNet handler class including different trained models, different modes for extracting the activation and causal and non-causal inferences

        Parameters
        ----------
        Inputs: 
            model: An scalar in the range [1,3] to select which pre-trained CRNN models to utilize. 
            mode: An string to determine the working mode. i.e. 'stream', 'realtime', 'online' and ''offline.
                'stream' mode: Uses the system microphone to capture sound and does the process in real-time. Due to training the model on standard mastered songs, it is highly recommended to make sure the microphone sound is as loud as possible. Less reverbrations leads to the better results.  
                'Realtime' mode: Reads an audio file chunk by chunk, and processes each chunck at the time.
                'Online' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then infers the parameters on interest using particle filtering.
                'offline' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then inferes the parameters on interest using madmom dynamic Bayesian network. This method is quicker that madmom beat/downbeat tracking.
            inference model: A string to choose the inference approach. i.e. 'PF' standing for Particle Filtering for causal inferences and 'DBN' standing for Dynamic Bayesian Network for non-causal usages.
            plot: A list of strings to plot. 
                'activations': Plots the neural network activations for beats and downbeats of each time frame. 
                'beat_particles': Plots beat/tempo tracking state space and current particle states at each time frame.
                'downbeat_particles': Plots the downbeat/meter tracking state space and current particle states at each time frame.
                Note that to speedup plotting the figures, rather than new plots per frame, the previous plots get updated. However, to secure realtime results, it is recommended to not plot or have as less number of plots as possible at the time.   
            threading: To decide whether accomplish the inference at the main thread or another thread. 
                  
            
        Outputs:
            A vector including beat times and downbeat identifier columns, respectively with the following shape: numpy_array(num_beats, 2).
    '''
    
    
    def __init__(self, model, mode='online', inference_model='PF', plot=[], thread=False):
        self.model = model
        self.mode = mode
        self.inference_model = inference_model
        self.plot= plot
        self.thread = thread
        if plot and thread:
            raise RuntimeError('Plotting cannot be accomplished in the threading mode')
        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)
        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, win_length=self.log_spec_win_length,
                             hop_size=self.log_spec_hop_length, n_bands=[24], mode = self.mode)
        if self.inference_model == "PF":                 # instantiating a Particle Filter decoder - Is Chosen for online inference
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=50, plot=self.plot, mode=self.mode)
        elif self.inference_model == "DBN":                # instantiating an HMM decoder - Is chosen for offline inference
            self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        else:
            raise RuntimeError('inference_model can be either "PF" or "DBN"')
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
        if self.mode == 'stream':
            self.stream_window = np.zeros(self.log_spec_win_length + 2 * self.log_spec_hop_length, dtype=np.float32)                                          
            self.stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                             channels=1,
                                             rate=self.sample_rate,
                                             input=True,
                                             frames_per_buffer=self.log_spec_hop_length,)
                                             
    def process(self, audio_path=None):   
        if self.mode == "stream":
            if self.inference_model != "PF":
                    raise RuntimeError('The infernece model should be set to "PF" for the streaming mode!')
            self.counter = 0
            while self.stream.is_active():
                self.activation_extractor_stream()  # Using BeatNet casual Neural network streaming mode to extract activations
                if self.thread:
                    x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                    x.start()
                    x.join()    
                else:
                    output = self.estimator.process(self.pred)
                self.counter += 1

                
        elif self.mode == "realtime":
            self.counter = 0
            self.completed = 0
            if self.inference_model != "PF":
                raise RuntimeError('The infernece model for the streaming mode should be set to "PF".')
            if audio_path != None:
                while self.completed == 0:
                    self.activation_extractor_realtime(audio_path) # Using BeatNet casual Neural network realtime mode to extract activations
                    if self.thread:
                        x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                        x.start()
                        x.join()    
                    else:
                        output = self.estimator.process(self.pred)  # Using particle filtering online inference to infer beat/downbeats
                    self.counter += 1
                return output
            else:
                raise RuntimeError('An audio object or file directory is required for the realtime usage!')
        
        
        elif self.mode == "online":
            if audio_path != None:
                preds = self.activation_extractor_online(audio_path)    # Using BeatNet casual Neural network to extract activations
            else:
                raise RuntimeError('An audio object or file directory is required for the online usage!')
            if self.inference_model == "PF":   # Particle filtering inference (causal)
                output = self.estimator.process(preds)  # Using particle filtering online inference to infer beat/downbeats
                return output
            elif self.inference_model == "DBN":    # Dynamic bayesian Network Inference (non-causal)
                output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
                return output
        
        
        elif self.mode == "offline":
                if self.inference_model != "DBN":
                    raise RuntimeError('The infernece model should be set to "DBN" for the offline mode!')
                if audio_path != None:
                    preds = self.activation_extractor_online(audio_path)    # Using BeatNet casual Neural network to extract activations
                    output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
                    return output
        
                else:
                    raise RuntimeError('An audio object or file directory is required for the offline usage!')
                

    def activation_extractor_stream(self):
        # TODO: 
        ''' Streaming window
        Given the training input window's origin set to center, this streaming data formation causes 0.084 (s) delay compared to the trained model that needs to be fixed. 
        '''   
        with torch.no_grad():
                    hop = self.stream.read(self.log_spec_hop_length)
                    hop = np.frombuffer(hop, dtype=np.float32)
                    self.stream_window = np.append(self.stream_window[self.log_spec_hop_length:], hop)             
                    if self.counter < 5:
                        self.pred = np.zeros([1,2])
                    else:
                        feats = self.proc.process_audio(self.stream_window).T[-1]
                        feats = torch.from_numpy(feats)
                        feats = feats.unsqueeze(0).unsqueeze(0)
                        pred = self.model(feats)[0]
                        pred = self.model.final_pred(pred)
                        pred = pred.detach().numpy()
                        self.pred = np.transpose(pred[:2, :])
                        

    def activation_extractor_realtime(self, audio_path):
        with torch.no_grad():
            if self.counter==0: #loading the audio
                if isinstance(audio_path, str):
                    self.audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
                else:
                    self.audio = audio_path
            if self.counter<(round(len(self.audio)/self.log_spec_hop_length)):
                if self.counter<2:
                    self.pred = np.zeros([1,2])   
                else:
                    feats = self.proc.process_audio(self.audio[self.log_spec_hop_length * (self.counter-2):self.log_spec_hop_length * (self.counter) + self.log_spec_win_length]).T[-1]
                    feats = torch.from_numpy(feats)
                    feats = feats.unsqueeze(0).unsqueeze(0)
                    pred = self.model(feats)[0]
                    pred = self.model.final_pred(pred)
                    pred = pred.detach().numpy()
                    self.pred = np.transpose(pred[:2, :])
            else:
                self.completed = 1


    def activation_extractor_online(self, audio_path):
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

