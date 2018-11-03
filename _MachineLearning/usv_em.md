---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: EM Algorithm
share: true
permalink: /MachineLearning/usv_em/
sidebar:
  nav: "MachineLearning"
---


# Introduction

In this section,we will introduce a new learning algorithm for density estimation, namely Expectation-Maximization (EN). Before we introduce what EM is, I will first talk about **mixture of Gaussian** model and build the intuition on that. 

Let's denote $\{x^{(1)},\dots,x^{(m)}\}$ the training dataset without labels. We assume that each data sample is associated with a class label, say $z^{(i)} \sim Multinomial(\phi)$. So $\phi_j = p(z^{(i)} = j)$. We aslo assume that data samples in each cluster is distributed as Gaussian. That is: $x^{(i)}\lvert z^{(i)}=j\sim\mathcal{N}(\mu_j,\Sigma_j)$. Then, we joint distribution as $p(x^{(i)},z^{(i)}) = p(x^{(i)}\lvert z^{(i)})p(z^{(i)})$. This looks like k means clustering. This is called **mixture of Gaussian**. We call $^{(i)}$ **latent variable** in this case, meaning it is invisible. 

The parameters that are to be optimized are $\phi,\mu,\Sigma$. The likelihood turns out to be:

$$\begin{align}
\ell(\phi,\mu,\Sigma) &= \sum\limits_{i=1}^m \log p(x^{(i)};\phi,\mu,\Sigma)\\
&= \sum\limits_{i=1}^m \log \sum\limits_{z^{(i)}=1}^k p(x^{(i)}\lvert z^{(i)};\mu,\Sigma)p(z^{(i)};\phi)
\end{align}$$

The standard way is to set its derivatives to zero and solve it with respect to each variable. **However, this cannot be solved in a closed form!**

let's take a look at this equation again. It is hard to solve because we have z variable there. $z^{(i)}$ indicates what class a data sample might belong to. We integrate this out, which makes it hard to calculate. If we knew what value z is, we can easily calcualte the likelihood as:

$$\ell(\phi,\mu,\Sigma) = \sum\limits_{i=1}^m \log p(x^{(i)})\lvert z^{(i);\mu,\Sigma} + \log p(z^{(i)};\phi)$$

We set the derivative of this to zero, and then we can update them as:

$$\phi_i = \frac{1}{m}\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\}$$

$$\mu_j = \frac{\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\} x^{(i)}}{\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}$$

$$\Sigma_j = \frac{\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\} (x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}$$

You can calculate them as practice. Note that the frist one is like the one in Gaussian Discriminative Analysis. For second one and third one, these formulas can be useful:

$$\frac{\partial x^TAx}{\partial x} = 2x^TA$$ iff A is symmetric and independent of x

$$\frac{\partial \log\lvert X\rvert}{\partial X} = X^{-1}$$

$$\frac{\partial a^TX^{-1}b}{\partial X} = -X^{-1}ab^Tx^{-1}$$

So we can see that if z is known, we can solve these parameters in one shot. **What it essentially means is that if we know the label for each data sample, we can find the proper portion, mean and variance for each cluster easily**. This sounds naturally true.

However, when z is unknown, we have to use iterative algorithm to find these parameters. EM comes to play!

EM, as name suggested, has two steps, expectation and maximization. In E step, we take a guess on what value z might be. In M step, it updates the model parameters based on the guess. Remember that if we know z, optimization becomes easier. 

1 E Step: for each i,j, $w_j^{(i)} = p(z^{(i)} = j\lvert x^{(i)}; \mu,\Sigma,\phi)$

2 M Step: 



