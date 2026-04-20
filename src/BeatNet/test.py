from BeatNet import BeatNet

estimator = BeatNet(1, mode='online', inference_model='PF', plot=['activations'], thread=False)


Output = estimator.process('/home/nikhil/moji/BeatNet/test/test_data/wav/Albums-AnaBelen_Veneo-02.wav')

print(Output)