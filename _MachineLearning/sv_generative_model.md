---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Generative Learning Algorithm
share: true
permalink: /MachineLearning/sv_generative_model/
sidebar:
  nav: "MachineLearning"
---

# 1 Discriminative Model

Algorithms try to directly classify a label for input such logstic regression and perceptron algorithm. The discriminative model does not have a concept of what the object might look like. They just classify. It cannot generate a new image based on the boundary.

# 2 Generative Model

Models fisrt try to learn each object might look like such as Bayesian method. Then, based on input, it gives a probability of the input being this class. It has the concepts on what the object might look like. It can generate a new image based on the past knowledge. 

With class prior, we can use Bayes rule to calculate the probability of being each class and then take the one with a bigger value. 

# 3 Gaussian Discriminant Analysis 

For a vector-values random variable Z:

$$Cov(Z) = E[(Z-E[Z])(Z-E[Z])^T] = E[ZZ^T - 2ZE[Z]^T + E[Z]E[Z]^T] $$

$$= E[ZZ^T] - 2E[Z]E[Z]^T + E[Z]E[Z]^T = E[ZZ^T] - E[Z]E[Z]^T$$

# 4 GDA and logistic regression

If $P(x\lvert y)$ is multivariate gaussian with shared covariance, then $P(y\lvert x)$ follows a logistic function. It means that GDA requires a strong assumption that data of each class can be modeled with a gaussian with shared covariance. However, GDA will fit better and train faster if assumptions are correct. 

On the other side, if assumption cannot be made, logistic regression is less sensitive. For example, Poisson can replace gaussian also leading to logistic regression. 

# 5 Naive Bayes

This is for learning discrete valued random variables like text classification. In text classification, a word vector is used for training. However, if we have 50000 words and try to model it as multinominal, then the dimension of parameter is $2^50000-1$, which is too large. Thus, we make **Naive Bayes Assumption:**

Each word is conditionally independent to each other based on given class. 

Then, we have:

$$P(x_1,...,x_50000\lvert y) = P(x_1\lvert y)P(x_2\lvert y,x_1)...P(x_50000\lvert y,x_1,x_2,...,x_49999) $$

$$= \prod\limits_{i=1}^{n} P(x_i\lvert y)$$

We apply **probability law of chain rule** for the first step and naive basyes assumption for the second step. 

After finding the max of **log joint likelihood**, which is:

$$\mathcal{L}(\phi_y,\phi_{j\lvert y=0},\phi_{j\lvert y=1}) = \prod\limits_{i=1}^{m} P(x^{(i)},y^{(i)}) $$

where $\phi_{j\lvert y=1} = P(x_j = 1 \lvert y = 1)$.

Then, we can use **Bayes Rule** to calculate $P(y=1\lvert x)$ and compare which is higher. 

**Ext**: In this case, we model $P(x_i\lvert y)$ as Bernouli since it is binary valued. That is, it can be either 'have that word' or 'not have that word'. Bernouli takes class label as input and models its probability but it has to binary. To deal with non-binary valued $x_i$, we can model it as Multinomial distribution, which can be parameterized with multiple classes. 

**Summary:** Naive Bayes is for discrete space. GDA is for continous space. We can alsway discretize it. 

# 6 Laplace smoothing

The above shwon example is generally good but will possibly fail where a new word which does exist in the past training samples appear in the coming email. In such case, it would cause $\phi$ for both classes to become zero because the models never see the word before. The model will fail to make prediction. 

This motivates a solution called **Laplace Smoothing**, which sets each parameter as:

$$\phi_j = \frac{\sum_{i=1}^{m} \mathbb{1}[z^{(i)}] + 1}{m+k}$$

where k is the number of classes. In reality, the Laplace smoothing does not make too much difference since it usually has all the words but it is good to have it here. 

# 7 Event Models

In generative setting, we should have class prior and likelihood for each class. For the likelihood model, it can be Bernoulli if it is binary or it can be Multinomial if it is multi-class. 