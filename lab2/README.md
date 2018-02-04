# MIT 6.S191 Lab 2: Deep Learning for Computer Vision
<p align="center" >
  <img src=https://github.com/aamini/introtodeeplearning_labs/raw/master/lab2/img/pneumothorax.png />
</p>

## Part 1: Handwritten Digit Classification
In the first portion of this lab, we will build and train a convolutional neural network (CNN) for classification of handwritten digits from the famous MNIST dataset. Our classes are the digits 0-9. This is a base example of building models for computer vision based on the `tf.layers` API.

## Part 2: Pneumothorax Detection from Human X-Ray Scans
This second section of the lab will introduce you to using a convolutional network to tackle a realistic dataset in medical diagnostics. Specifically, we use the ChestXRay dataset, a set of X-ray images labeled with corresponding diagnoses to build a model that classifies patients with a Pneumothorax (ie. dropped or collapsed lung). Additionally you will implement a state of the art feature visualization technique to answer questions like: *why did the network make this decision?* or *where is the network looking?*

<p align="center">
  <img src=https://github.com/aamini/introtodeeplearning_labs/raw/master/lab2/img/heatmap_example.png />
</p>


