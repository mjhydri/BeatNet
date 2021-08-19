# BeatNet
This repository contains the source code and additional documentation of "BeatNet" music time analizer which is 2021's state of the art online joint beat/downbeat/meter tracking system. The original ISMIR paper can be found on the following link: https://arxiv.org/abs/2108.03576

In addition to online beat/downbeat/meter tracking, we added madmom's DBN beat/downbeat inference model from offline usages. Note that for such purpose we still utilize BeatNet's neural network rather than that of Madmom which leads to better performance and significantly faster results.
___________________________________________________________________
Installation: 
pip install git+https://github.com/mjhydri/BeatNet
___________________________________________________________________
Usage example:
estimator = BeatNet(1,'PF')
beats,downbeats =estimator.process("<music file directory>")
___________________________________________________________________
  
In order to demonstrate the performance of the system for different beat/donbeat tracking difficulties, here are three demo examples :

1: Easy song
<iframe width="560" height="315" src="https://www.youtube.com/embed/XsdA4AATaUY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

2: Medium song
<iframe width="560" height="315" src="https://www.youtube.com/embed/GuW8C5xuWbQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

3: Veteran song
<iframe width="560" height="315" src="https://www.youtube.com/embed/dFbFGMs9CA4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

