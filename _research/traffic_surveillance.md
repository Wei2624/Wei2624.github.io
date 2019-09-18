---
layout: single
mathjax: true
toc: true
toc_sticky: true
author_profile: true
title: Traffic Surveillance Research
share: false
permalink: /research/traffic_surveillance/
---

This work is completed for NVIDIA AI Smart City Competition 2018. 

# Overview

Traffic surveillance system has been one of the most intriguing aspects of smart city application. In the context of smart city management, it is important to monitor traffic situations using advanced algorithm. In particular, we are interested in traffic speed estimation and traffic anomaly detection such as the event of car stopping. 

In speed estimation task, we propose to first use Mask-RCNN to generate bounding boxes for vehicles on road. Then, we apply a multi-object tracker for tracking each detected vehicle. Within a predefined frame interval $T_1$, we reconstruct 3D information for each bounding box to estimate speed. Such a pipeline can be viewed below.

![Pipeline](/_research/images/traffic_surveillance_1.png)

In anomaly detection task, we propose to frist use pretrained VGG-16 network to extract visual features. In parallel with feature extraction, we also extract temporal visual features by implementing improved dense trajectory algorithm. Then, we encode both extracted features into lower dimension space and concatenate into one single vector. Last but not least, we train a support vector machine model with such a vector resulted from last step. The full pipeline can be viewed below. 

![Pipeline](/_research/images/traffic_surveillance_2.pdf)

An video demo can be watched here. In the demo, green box is resulted from tracking algorithm and yellow box is resulted from detection algorithm. 

{% include video id="pMRB7cjQabc" provider="youtube" %}

# Contributions

In this paper, I am responsible for implementing Mask-RCNN and the entire pipeline of anomaly detection task, including training and testing. In addition, I am also respobsible for writing the corresponding parts in the paper. 

|:-----:|:-------:|
|Platform|Ubuntu 16.04|
|Programming Language|C++,Python,Matlab|
|Deep Learning Framework|Tensorflow|
|Research Area|Computer Vision|
