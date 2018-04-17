# MIT 6.S191 Lab 1: Intro to TensorFlow and Music Generation with RNNs

![alt text](https://github.com/aamini/introtodeeplearning_labs/raw/master/lab1/img/music_waveform.png)
## Part 1: Intro to TensorFlow
TensorFlow is a software library extensively used in machine learning. Here we'll learn how computations are represented and how to define simple neural networks in TensorFlow. In this section you will learn the basic of Tensorflow computational graphs, Sessions, and the new imperitive version of Tensorflow: [Eager](https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html).

## Part 2: Music Generation with RNNs
In this portion of the lab, we will play around with building a Recurrent Neural Network (RNN) for music generation. We will be using the MIDI music toolkit to create a dataset of music files and build a model that captures the long term dependencies in musical notes. Finally, we will sample from this model to generate brand new music that has never been heard before!

### Troubleshooting

In Part 2 of this lab, while running the very first command (`pip install python-midi`), if you get the following error

```
Command 'lsb_release -a' returned non-zero exit status 1
```

you can resolve it by removing the `lsb_release` file. It's not really required by `pip`. To do so, add the following command to the notebook before `pip install python-midi` and run it.

```
!mv /usr/bin/lsb_release /usr/bin/lsb_release.bkup
```
