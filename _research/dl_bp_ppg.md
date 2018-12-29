---
layout: single
mathjax: true
toc: true
toc_sticky: true
author_profile: true
title: Deep Learning for Blood Pressure from Photoplethysmography 
share: false
permalink: /research/dl_bp_ppg/
---

# Overview

In medical field, blood pressure is a crucial indicator for fatal diseases such as heart disease and stroke. Continuous blood pressure measurement is also found to be an informative measurement for chronic disease. To measure blood pressure, an operator using sphygmomanometer is usually required. This traditional method is time-consuming and cannot continuously montior blood pressure of a patient. Thus, a pressure-cuff-free and operator-free method for blood pressure measurement is desired. 

We propose to use Photoplethysmography (PPG) signal to continuously estimate blood pressure using a deep convolutional neural network. The pipeline can be viewed below.

![Pipeline](/_research/images/dl_bp_ppg_1.png)

The deep convolutional neural network consists of convolutional layer (Conv), local response normalization (LRN), rectifier linear unit (Relu) and Droptout. It can be visualized below. 

![Model](/_research/images/dl_bp_ppg_2.png)

# Contributions

In this work, I am fully responsible for algorithm designing and testing. After dataset cleaning is performed by my colleague, I train the proposed model using 80% of cleaned dataset and test the trained model using the rest 20% of cleaned dataset. In addition, I am also fully responsible for writing the paper. 