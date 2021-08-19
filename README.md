# BeatNet
This repository contains the source code and additional documentation of "BeatNet" music time analizer which is 2021's state of the art online joint beat/downbeat/meter tracking system. The original ISMIR paper can be found on the following link: https://arxiv.org/abs/2108.03576

In addition to online beat/downbeat/meter tracking, we added madmom's DBN beat/downbeat inference model from offline usages. Note that for such purpose we still utilize BeatNet's neural network rather than that of Madmom which leads to better performance and significantly faster results.
___________________________________________________________________
Installation command: 
pip install git+https://github.com/mjhydri/BeatNet
___________________________________________________________________
Usage example:
estimator = BeatNet(1,'PF') 

beats,downbeats =estimator.process("music file directory")
___________________________________________________________________
  
In order to demonstrate the performance of the system for different beat/donbeat tracking difficulties, here are three demo examples :

1: Song Difficulty: Easy
  
  
[![Easy song](https://img.youtube.com/vi/XsdA4AATaUY/0.jpg)](https://www.youtube.com/watch?v=XsdA4AATaUY)
  



2: Song difficulty: Medium
  
  [![Easy song](https://img.youtube.com/vi/GuW8C5xuWbQ/0.jpg)](https://www.youtube.com/watch?v=GuW8C5xuWbQ)
  




3: Song difficulty: Veteran
  
  [![Easy song](https://img.youtube.com/vi/dFbFGMs9CA4/0.jpg)](https://www.youtube.com/watch?v=dFbFGMs9CA4)


