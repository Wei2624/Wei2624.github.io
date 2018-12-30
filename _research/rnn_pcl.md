---
layout: single
mathjax: true
toc: true
toc_sticky: true
author_profile: true
title: Enhanced Recurrent Neural Network Semantic Labeling with Point Cloud Processing
share: false
permalink: /research/rnn_pcl/
---

This work is performed at Columbia Robotics Lab and advised by Professor Peter Allen. This is an ongoing work and only part of results are shown here. 

# Overview

Recently, the dramatic explosion and success on machine learning and computer vision have stimulated a new field of robotics research into the next phase. In particular, semantic mapping in robotics has been proven a promising direction for grasping and manipulation. However, real-world samples are generally scarce and expensive to collect. Thus, training only on simulation dataset can give noisy labels when deployed in real world. 

In this work, we combine domain randomization and a point cloud post-processing step to enhance semantic labeling of a table-top scene. The whole pipeline involves two branches. The first branch takes a RGB image and a depth image as input and feed them into a deep convolutional neural network. The deep convolutional neural network contains a VGG-16 network, convolutional transpose layers and a recurrent neural network. It will output a per-pixel semantic labeling map with the same size of input. Each pixel represents a prediction class among a predefined class set. The second branch takes a depth image as input and generate a point cloud from depth image. Then a 3D clustering algorithm is performed to segment point cloud. The final output is resulted from the combination of prediction map and clustering results. The whole pipeline can be viewed below. 

![Pipeline](/_research/images/rnn_pcl_1.png)

Part of results in image can be viewed below.

![Image Demo](/_research/images/rnn_pcl_2.png)

An video demo can be viewed below. 

{% include video id="T7zllZbtm2A" provider="youtube" %}


# Contributions

In this project, I have been working with Professor Peter Allen since the beginning. I am fully responsible for building and testing the entire pipeline. First, I reproduced an algorithm from the paper called "DA-RNN: Semantic Mapping with Data Associated Recurrent Neural Networks." Secondly, I dived into sveral simulation tools for generating simulation dataset. Last but not least, I tested many promising algorithms to solve the table-top semantic mapping task. 

|:-----:|:-------:|
|Platform|Ubuntu 16.04, ROS|
|Programming Language|C++,Python|
|Deep Learning Framework|Tensorflow|
|Research Area|Computer Vision, Robotics|
