---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Nerual Networks
share: true
permalink: /MachineLearning/usv_kmeans/
sidebar:
  nav: "MachineLearning"
---

# Introduction

In supervised learning, we are always given all the labels/ground truth in training phase. This makes it supervised property. Note that in general that supervised learning assumes that each sample is i.i.d. in the training and testing samples. 

In unsupervised learning, we are not given any labels or ground truth for training. We are simply taking input into training model. We call it **unsupervised learning**. 

# K-means Clustering Algorithm

K-means clustering algorithm is a standard unsupervised learning algorithm. K-means usually will generate K clusters based on the distance of data point and cluster mean. On the other hand, knn clustering algorithm usually will return clusters with k samples for each cluster. Keep in mind that there is no label or ground truth required. 

We are given a training set $\{x^{(1)},x^{(2)},\dots,x^{(m)}\}$ where $x^{(i)}\in \mathbb{R}^n$. Then we can define K-means clustering algorithm as:

1 Initialize **cluster centroids** $\mu_1,\mu_2,\dots,\mu_k\in mathbb{R}^n$ rrandomly

2 Repeat until convergence:

&nbsp;&nbsp;&nbsp;&nbsp; For every i, set $c^i = \arg\min_j\lvert\lvert x^i - \mu_j\rvert\rvert^2$

&nbsp;&nbsp;&nbsp;&nbsp; For each j, set $\mu_j = \frac{\sum_{i=1}^m\mathbb{1}[c^i=j]x^i}{\sum_{i=1}^m\mathbb{1}[c^i=j]}$

K is the parameter that we need to predefine. This is called **parametric laerning**. After selecting K, we can ramdomly pick up K samples to be our K centroids. Surly, we can use some other way to initialize them. 

In the loop, we repeatedly execute two steps. First, assign each training sample to the cloest centroid $\mu_j$. Second, moving each cluster centroid to the mean of samples assigned to it. The figure below can show this process. 

![K Means](/images/cs229_usv_keams.png)

Plot a is the plot of samples. Plot b is samples with centroids. The rest plots show the training process. 

A natural question to ask is: Is the k-means algorithm guaranteed to converge?

The answer is yes. 

We can define the loss function to be:

$$ J(c,\mu) = \sum\limits_{i=1}^m\lvert\lvert x^i - \mu_{c^i}\rvert\rvert^2$$

We can show that k-means algorithm is exactly coordinate descent on J. Remember that coordinate descent is to minimize the cost function with respect to one of the variables while holding the others constant. Thus, we can always find out that J, c and $\mu$ will always converge. 

Since J is non-convex function, coordinate descent is not guaranteed to converge to the global minimum. Rather, it will always converge to local mimimum. To avoid this, we can run k-means several times with different initilizations and choose the best in terms of J. 
