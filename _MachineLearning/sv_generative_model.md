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

The discriminative model is the model where algorithms try to directly classify a label for input such logstic regression and perceptron algorithm. The discriminative model does not have a concept of what the object might look like. They just classify. It cannot generate a new image based on the class. 

Formally, it is $p(y\lvert x;\theta)$ where p can be any classification model such as logistic regression model. 

# 2 Generative Model

On the other hand, the generative model is the models that fisrt try to learn what each object might look like. Then, based on input, it gives a probability of the input being this class. It has the concepts on what the object might look like. It can generate a new image based on the past knowledge. 

The classical example is naive Bayes classifier. In this case, we have a class prior. With class prior, we can use Bayes rule to calculate the probability of being each class and then take the one with a bigger value. Meanwhile, with a certain prior, we can generate features based on the chosen prior. This is generative process. 

# 3 Gaussian Discriminant Analysis 

Gaussian discriminant analysis (GDA) model is a generative model where $p(x\lvert y)$ is a multi-variate Gaussian. So I will start talking about multi-veriate Gaussian. 

## 3.1 The Multivariate Normal Distribution

In multivariate normal distribution, a random variable is vector-valued in $\mathbb{R}^n$ where n is the number of dimensionality. Thus, multivariate Gaussian has mean vector $\mu\in \mathbb{R}^n$ and covariance matrix $\Sigma\in\mathbb{R}^{n\times n}$ where $\Sigma is sysmetric and postive semi-definite. The density is:

$$p(x;\mu,\Sigma) = \frac{1}{(2\pi)^{n/2}\lvert \Sigma\rvert^{1/2}}\exp\bigg(-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu)\bigg)$$

As said, the expectation is $\mu$.

The covariance for a vector-values random variable Z:

$$Cov(Z) = E[(Z-E[Z])(Z-E[Z])^T] = E[ZZ^T - 2ZE[Z]^T + E[Z]E[Z]^T] $$

$$= E[ZZ^T] - 2E[Z]E[Z]^T + E[Z]E[Z]^T = E[ZZ^T] - E[Z]E[Z]^T$$

An example of plot of density function with zero mean but different covariance can be shwon below. 

![Multivariate Gaussian](/images/cs229_gen_mul_gau.png)

In this example, we have covariance frome left from right:

$$\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}; \Sigma = \begin{bmatrix} 1 & 0.5 \\ 0.5 & 1 \end{bmatrix}; \Sigma = \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix}$$

# 4 GDA and logistic regression

## 4.1 GDA

Let's talk about binary classification problem again. We can use multivariate Gaussian to model $p(x\lvert y)$. Put all together, we have:

$$y \sim Bernoulli(\phi)$$

$$x\lvert y=0 \sim \mathcal{N}(\mu_0,\Sigma)$$

$$x\lvert y=1 \sim \mathcal{N}(\mu_1,\Sigma)$$

where $\phi, \mu_0,\mu_1,\Sigma$ is the parameters that we want to find out. Note that although we have different mean for different classes, we have shared covariance between different classes. Then, the log likelihood of data is:

$$\begin{align}
\ell(\phi,\mu_0,\mu_1,\Sigma) &= \log \prod_{i=1}^m p(x^{(i)}, y^{(i)};\phi,\mu_0,\mu_1,\Sigma) \\
&= \log \prod_{i=1}^m p(x^{(i)}\lvert y^{(i)};\mu_0,\mu_1,\Sigma) p(y^{(i)};\phi)
\end{align}$$
 
By maximizing the above with respect to the parameters, we have:

$$\begin{align}
\phi &= \frac{1}{m}\sum\limits_{i=1}^m \mathbb{1}\{y^{(i)}=1\} \\
\mu_0 &= \frac{\sum_{i=1}^m\mathbb{1}\{y^{(i)}=0\}x^{(i)}}{\sum_{i=1}^m\mathbb{1}\{y^{(i)}=0\}} \\
\mu_1 &= \frac{\sum_{i=1}^m\mathbb{1}\{y^{(i)}=1\}x^{(i)}}{\sum_{i=1}^m\mathbb{1}\{y^{(i)}=1\}} \\
\Sigma &= \frac{1}{m}\sum\limits_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
\end{align}$$

The results can be shown as:

![GDA Learning](/images/cs229_gen_gda_learn.png)

Note that we have shared covariance so the shape of two contours are the same but the means are different. On the boundary line, we have probability of 0.5 for each class. 

## 4.2 GDA and Logistic Regression

If $P(x\lvert y)$ is multivariate gaussian with shared covariance, then $P(y\lvert x)$ follows a logistic function. If $P(x\lvert y)$ is Possion with different $\lambda$, then $P(y\lvert x)$ also follows a logistic function. It means that GDA requires a strong assumption that data of each class can be modeled with a gaussian with shared covariance. However, GDA will fit better and train faster if assumptions are correct. 

On the other side, if assumption cannot be made, logistic regression is less sensitive. For example, Poisson can be replaced with gaussian also leading to logistic regression. 

# 5 Naive Bayes

In GDA, random variables are supposed to be continuous-valued. In Naive Bayes, it is for learning discrete valued random variables like text classification. Text classification is to classify text based on the words in it to a binary class. In text classification, a word vector is used for training. A word vector is like a dictionary. The length of the vector is the number of words. A word is represented by a 1 on certain position and elsewhere with 0's in the vector. 

However, this might not work. Say, if we have 50000 words and try to model it as multinominal, then the dimension of parameter is $2^50000-1$, which is too large. Thus, to solve it, we make **Naive Bayes Assumption:**

Each word is conditionally independent to each other based on given class. 

Then, we have:

$$P(x_1,...,x_50000\lvert y) = P(x_1\lvert y)P(x_2\lvert y,x_1)...P(x_50000\lvert y,x_1,x_2,...,x_49999) $$

$$= \prod\limits_{i=1}^{n} P(x_i\lvert y)$$

We apply **probability law of chain rule** for the first step and naive basyes assumption for the second step. 

After finding the max of **log joint likelihood**, which is:

$$\mathcal{L}(\phi_y,\phi_{j\lvert y=0},\phi_{j\lvert y=1}) = \prod\limits_{i=1}^{m} P(x^{(i)},y^{(i)}) $$

where $\phi_{j\lvert y=1} = P(x_j = 1 \lvert y = 1)$, $\phi_{j\lvert y=0} = P(x_j = 1 \lvert y = 0)$ and $\phi_y = p(y=1)$. Those are the parameters that we want to learn. 

We can find the derivative and solve them:

$$\begin{align}
\phi_{j\lvert y=1} &= \frac{\sum_{i=1}^m \mathbb{1}\{x_j^i = 1 \text{and} y^i = 1\}}{\sum_{i=1}^m \mathbb{1}\{y^i = 1\}} \\
\phi_{j\lvert y=0} &= \frac{\sum_{i=1}^m \mathbb{1}\{x_j^i = 1 \text{and} y^i = 0\}}{\sum_{i=1}^m \mathbb{1}\{y^i = 0\}} \\
\phi_y &= \frac{\sum_{i=1}^m \mathbb{1}\{y^i = 1\}}{m} \\
\end{align}$$

To predict for a new sample, we can use **Bayes Rule** to calculate $P(y=1\lvert x)$ and compare which is higher. 

$$\begin{align}
p(y=1\lvert x) &= \frac{p(x\lvert y=1)p(y=1)}{p(x)} \\
 &= \frac{p(y=1)\prod_{j=1}^n p(x_j\lvert y=1)}{p(y=0)\prod_{j=1}^n p(x_j\lvert y=0) + p(y=1)\prod_{j=1}^n p(x_j\lvert y=1)} \\
\end{align}$$

**Ext**: In this case, we model $P(x_i\lvert y)$ as Bernouli since it is binary valued. That is, it can be either 'have that word' or 'not have that word'. Bernouli takes class label as input and models its probability but it has to binary. To deal with non-binary valued $x_i$, we can model it as Multinomial distribution, which can be parameterized with multiple classes. 

**Summary:** Naive Bayes is for discrete space. GDA is for continous space. We can alsway discretize it. 

# 6 Laplace smoothing

The above shwon example is generally good but will possibly fail where a new word which does exist in the past training samples appear in the coming email. In such case, it would cause $\phi$ for both classes to become zero because the models never see the word before. The model will fail to make prediction. 

This motivates a solution called **Laplace Smoothing**, which sets each parameter as:

$$\begin{align}
\phi_{j\lvert y=1} &= \frac{\sum_{i=1}^m \mathbb{1}\{x_j^i = 1 \text{and} y^i = 1\}+1}{\sum_{i=1}^m \mathbb{1}\{y^i = 1\}+2} \\
\phi_{j\lvert y=0} &= \frac{\sum_{i=1}^m \mathbb{1}\{x_j^i = 1 \text{and} y^i = 0\}+1}{\sum_{i=1}^m \mathbb{1}\{y^i = 0\}+2} \\
\phi_j &= \frac{\sum_{i=1}^{m} \mathbb{1}[z^{(i)}] + 1}{m+k} \\
\end{align}$$

where k is the number of classes. In reality, the Laplace smoothing does not make too much difference since it usually has all the words but it is good to have it here. 