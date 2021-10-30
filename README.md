# BeatNet
This repository contains the source code and additional documentation of "BeatNet" music time analizer which is 2021's state of the art online joint beat/downbeat/meter tracking system. The original ISMIR paper can be found on the following link: https://arxiv.org/abs/2108.03576

In addition to online beat/downbeat/meter tracking, we added madmom's DBN beat/downbeat inference model from offline usages. Note that for such purpose we still utilize BeatNet's neural network rather than that of Madmom which leads to better performance and significantly faster results.

System Input:
-------------
Raw audio waveform 

System Output:
--------------
A vector including beats and downbeats columns, respectively with the following shape: numpy_array(num_beats, 2).

Installation command:
---------------------
Approach #1: Installing binaries from the pypi website:

pip install BeatNet


Approach #2: Installing directly from the Git repository:

pip install git+https://github.com/mjhydri/BeatNet

Usage example:
--------------
From BeatNet.BeatNet import BeatNet

estimator = BeatNet(1) 

Output = estimator.process("music file directory", inference_model= 'PF', plot = True)
  
A brief video tutorial of the system (Overview):
------------------------------------------

[![Easy song](https://img.youtube.com/vi/xOX74cXQKrY/0.jpg)](https://youtu.be/xOX74cXQKrY)

___________________________________________________________________

  
In order to demonstrate the performance of the system for different beat/donbeat tracking difficulties, here are three video demo examples :

1: Song Difficulty: Easy
  
  
[![Easy song](https://img.youtube.com/vi/XsdA4AATaUY/0.jpg)](https://www.youtube.com/watch?v=XsdA4AATaUY)
  



2: Song difficulty: Medium
  
  [![Easy song](https://img.youtube.com/vi/GuW8C5xuWbQ/0.jpg)](https://www.youtube.com/watch?v=GuW8C5xuWbQ)
  




3: Song difficulty: Veteran
  
  [![Easy song](https://img.youtube.com/vi/dFbFGMs9CA4/0.jpg)](https://www.youtube.com/watch?v=dFbFGMs9CA4)


