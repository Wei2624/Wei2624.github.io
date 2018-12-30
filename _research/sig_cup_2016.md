---
layout: single
mathjax: true
toc: true
toc_sticky: true
author_profile: true
title: IEEE Signal Processing Cup 2016 Global Competition
share: false
permalink: /research/sig_cup_2016/
---

This work was performed at University of Waterloo where I was supervised by Professor Xueming(Sherman) Shen for undergaduate study. 

# Overview

Electrical network frequency (ENF) has been considered as evidence for location forensic. With ENF signal, we can potentially determine a location. This is done by extracting features from ENF signal and classify it based on prior knowledge of different grids. 

In this report, we propose a framework composing of two parts. The first part is to convert ENF signal from time domain to frequency domain. With ENF signal from both time domain and frequency domain, we select features using cross-validation method. The second part is to use multi-classes support vector machine algorithm to classify into a location gird based on the selected features. 

The features that are considered are:

![Features Considered](/_research/images/sig_cup_2016_1.png)

The feature selection results from cross-validation method can be found below.

![Features Considered](/_research/images/sig_cup_2016_2.png)


# My Contributions

In this work, I mainly focus on two parts. First of all, I was fully responsible for designing and building the hardware and the firmware to collect raw power signal from power jack at Waterloo, ON., Canada. The collected data were used as training and testing data. Second, I was also responsible for implementing the training and testing code for multi-classes support vector machine. The results were ranked top 10 in IEEE Signal Processing Cup 2016 Global Competition. 

|:-----:|:-------:|
|Platform|Windows 7, Mac OS|
|Programming Language|Matlab|
|Deep Learning Framework|scikit-learn|
|Research Area|Digital Signal Processing, Machine Learning|