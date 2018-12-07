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

In this section,we will introduce a new learning algorithm for density estimation, namely Expectation-Maximization (EM) algorithm. Before we introduce what EM is, I will first talk about **mixture of Gaussian** model and build the intuition on that. 

Let's denote $\{x^{(1)},\dots,x^{(m)}\}$ the training dataset without labels. We assume that each data sample is associated with a class label, say $z^{(i)} \sim Multinomial(\phi)$. So $\phi_j = p(z^{(i)} = j)$. We aslo assume that data samples in each cluster is distributed as Gaussian. That is: $x^{(i)}\lvert z^{(i)}=j\sim\mathcal{N}(\mu_j,\Sigma_j)$. Then, we have joint distribution as $p(x^{(i)},z^{(i)}) = p(x^{(i)}\lvert z^{(i)})p(z^{(i)})$. This looks like k means clustering. This is called **mixture of Gaussian**. We call $z^{(i)}$ **latent variable** in this case, meaning it is invisible. 

The parameters that are to be optimized are $\phi,\mu,\Sigma$. The likelihood turns out to be:

$$\begin{align}
\ell(\phi,\mu,\Sigma) &= \sum\limits_{i=1}^m \log p(x^{(i)};\phi,\mu,\Sigma)\\
&= \sum\limits_{i=1}^m \log \sum\limits_{k=1}^K p(x^{(i)}\lvert z^{(i)}=k;\mu_k,\Sigma_k)p(z^{(i)}=k;\phi)
\end{align}$$

The standard way is to set its derivatives to zero and solve it with respect to each variable. **However, this cannot be solved in a closed form!**

let's take a look at this equation again. It is hard to solve because we have z variable there. $z^{(i)}$ indicates what class a data sample might belong to. We have to integrate this out, which makes it hard to calculate. If we knew what value z is at the beginning, we can easily calcualte the likelihood as:

$$\ell(\phi,\mu,\Sigma) = \sum\limits_{i=1}^m \log p(x^{(i)}\lvert z^{(i)};\mu_{z^{(i)},\Sigma_{z^{(i)}) + \log p(z^{(i)};\phi)$$

We set the derivative of this to zero, and then we can update them as:

$$\phi_j = \frac{1}{m}\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\}$$

$$\mu_j = \frac{\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\} x^{(i)}}{\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}$$

$$\Sigma_j = \frac{\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\} (x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\sum\limits_{i=1}^m \mathbb{1}\{z^{(i)}=j\}}$$

You can calculate them as practice. Note that the frist one is like the one in [Gaussian Discriminative Analysis](https://wei2624.github.io/MachineLearning/sv_generative_model/). For second one and third one, these formulas can be useful:

$$\frac{\partial x^TAx}{\partial x} = 2x^TA$$ iff A is symmetric and independent of x

$$\frac{\partial \log\lvert X\rvert}{\partial X} = X^{-T}$$

$$\frac{\partial a^TX^{-1}b}{\partial X} = -X^{-T}ab^Tx^{-T}$$

Again, the proofs of these can be found in [Gaussian Discriminative Analysis](https://wei2624.github.io/MachineLearning/sv_generative_model/) section. 

So we can see that if z is known, we can solve these parameters in one shot. **What it essentially means is that if we know the label for each data sample, we can find the proper portion, mean and variance for each cluster easily**. This sounds naturally true.

However, when z is unknown, we have to use iterative algorithm to find these parameters. **EM comes to play!**

If we do not know something, we can take a guess on it. That's what EM does for us. EM, as name suggested, has two steps, expectation and maximization. In E step, we take a "soft" guess on what value z might be. In M step, it updates the model parameters based on the guess. Remember that if we know z, optimization becomes easier. 

1 E Step: for each i,j, $w_j^{(i)} = p(z^{(i)} = j\lvert x^{(i)}; \mu,\Sigma,\phi)$

2 M Step: update parameters:

&nbsp;&nbsp;&nbsp;&nbsp; $$\phi_j = \frac{1}{m}\sum\limits_{i=1}^m w_j^i$$

&nbsp;&nbsp;&nbsp;&nbsp; $$\mu_j = \frac{\sum\limits_{i=1}^m w_j^i x^{(i)}}{\sum\limits_{i=1}^m w_j^i}$$

&nbsp;&nbsp;&nbsp;&nbsp; $$\Sigma_j = \frac{\sum\limits_{i=1}^m w_j^i (x^{(i)}-\mu_j)(x^{(i)}-\mu_j)^T}{\sum\limits_{i=1}^m w_j^i}$$

How do we calculate the E step? In E step, we calculate z  by conditioning on current setting of all the parameters, which is the posterior. By using Bayes rule, we have:

$$o(z^{(i)}=j\lvert x^{(i)};\phi,\mu,\Sigma) = \frac{p(x^{(i)}\lvert z^{(i)};\mu,\Sigma)p(z^{(i)};\phi)}{\sum\limits_{k=1}^K p(x^{(i)}\lvert z^{(i)}=k;\mu,\Sigma)p(z^{(i)}=k;\phi)}$$

So $w_j^i$ is the soft guess for $z^{(i)}$, indicating that how likely sample i belongs to class j. This is also reflected by the updating euqation where instead of indicator funciton, we have a probablity to sum up. Indicator, on the other hand, is called hard guess. Similar to K means clustering, this algorithm is also susceptible to local optima, so initilizing paramsters several times might be a good idea. 

This shows us how EM generally works. I use **mixture of Gaussain** as an example. Next, I will show why EM works. 


# The EM algorithm

We have talked about EM algorithm by introducing mixture of Gaussian as an example. Now, we want to analyze EM in mathematical way.  How does EM work? Why does it work in general? Does it guarantee to converge? 

## Jensen's inequality

Let's first be armed with the definition of convex function and Jensen's inequality. 

**Definition:** A function f is a convex function if $f^{\ast\ast}(x)\geq 0$ for $x\in\mathcal{R}$ or its hessian H is positve semi-definite if f is a vector-values function. When both are strictly larger than zero, we call f a strictly convex function. 

**Jensen's inequality:** Let f be a convex function, and let X be a random variable. Then:

$$E[f(X)] \geq f(E[X])$$

Moreover, if f is strictly convex, $E[f(X)] = f(E[X])$ is true iff X is constant. 

What that means is that with a convex function f, and two points on X-axis with each probability of 0.5 to be selected. We can see that the function value of expected X is less or equal than the expected function value on two points. Such a concept can be visualized in below.

![EM Jensen's Inequality](/images/cs229_usv_em_jensen.png)

**Note** This also holds true for concave function since concave funciton is just the reverse of convex function. The inequality is also reversed. 

## EM Algorithm

With m training samples and a latent variable for each sample, we are trying to maxmimize the likelikhood defined as:

$$\ell(\theta) = \sum\limits_{i=1}^m\log p(x;\theta) = \sum\limits_{i=1}^m\log \sum\limits_z p(x,z;\theta)$$

As discussed, it is hard to calculate the derivative of this equation unless z is observed. 

EM comes here to solve this issue as shown in last section. Essentially, E-step tries to set a lower bound on loss function, and M-step tries to optimize parameters based on the bound. 

Let's define a distribution on class label for each sample i. We denote $q_i$ be some distribution where $\sum_z q_i(z)=1$. Then, we can extend the likelihood as:

$$\begin{align}
\sum\limits_i \log p(x^{(i)};\theta) &= \sum\limits_i\log\sum\limits_{z^i} p(x^{(i)},z^{(i)};\theta)\\
&= \sum\limits_i\log\sum\limits_{z^i} q_i(z^{(i)}) \frac{p(x^{(i)},z^{(i)};\theta)}{q_i(z^{(i)})}\\
&\geq \sum\limits_i\sum\limits_{z^i} q_i(z^{(i)}) \log\frac{p(x^{(i)},z^{(i)};\theta)}{q_i(z^{(i)})}
\end{align}$$

Last step is from Jensen's inequality where f is log function. Log function is a concave function and 

$$\sum\limits_{z^i} q_i(z^{(i)}) \bigg[\frac{p(x^{(i)},z^{(i)};\theta)}{q_i(z^{(i)})}\bigg] = \mathbb{E}_{z^i\sim q_i(z^{(i)})}\bigg[\frac{p(x^{(i)},z^{(i)};\theta)}{q_i(z^{(i)})}\bigg]$$

is the expectation over the random variable defined in the square bracket. **Thus, by doing this, we set the lower bound of joint lpg-likelihood.**

How do we choose $q_i$? There are many ways to define this distribution as long as it is a simplex. So How do we select from them? When we fix $\theta$, we always to make the bound as tight as possible. When is it the tightest? **It is when inequality becomes equality!**

How do we make it equal? Remember that for convex/concave function, Jensen's inequality holds with equality iff the random variable becomes constant. In this case, 

$$ \frac{p(x^{(i)},z^{(i)};\theta)}{q_i(z^{(i)})} = c$$

for some c that does not denpend on $z^i$. We know that if :

$$q_i(z^{(i)}) \propto p(x^{(i)},z^{(i)};\theta)$$

then, we can always have a constant as a result. In this case, we can select:

$$\begin{align}
q_i(z^{(i)}) &= \frac{p(x^{(i)},z^{(i)};\theta)}{\sum_z p(x^{(i)},z;\theta)}\\
&= \frac{p(x^{(i)},z^{(i)};\theta)}{p(x^{(i)};\theta)}\\
&= p(z^{(i)}\lvert x^{(i)};\theta)
\end{align}$$

Plugging the RHS in first line will always give a constant for the random variable. The last line just shows that $q_i(z^{(i)})$ is just the posterior of z based on data sample and the parameters. 

Let's put them all together. So we have:

1 E-step: for each i, set:

$$q_i(z^{(i)}) = p(z^{(i)}\lvert x^{(i)};\theta)$$

2 M-step: update parameters as :

$$\theta = \arg\max_{\theta} \sum\limits_i\sum\limits_{z^i} q_i(z^{(i)}) \log\frac{p(x^{(i)},z^{(i)};\theta)}{q_i(z^{(i)})}$$

The question now is that does this always converge? We want to prove that $\ell(\theta^t)\leq\ell(\theta^{t+1})$. So we have:

$$\begin{align}
\ell(\theta^{t+1}) &\geq \sum\limits_i\sum\limits_{z^i} q_i^t(z^{(i)}) \log\frac{p(x^{(i)},z^{(i)};\theta^{t+1})}{q_i^t(z^{(i)})}\\
&\geq \sum\limits_i\sum\limits_{z^i} q_i^t(z^{(i)}) \log\frac{p(x^{(i)},z^{(i)};\theta^{t})}{q_i^t(z^{(i)})}\\
&= \ell(\theta^t)
\end{align}$$

The first inequality is from Jensen's inequality which holds for all possible q and $\theta$. The second is because $\theta^{t+1}$ is updated from $\theta^t$ towards the maximization of this likelihood. The last holds because q is chosen in a way that Jensen's inequality holds with equality. 

This shows EM algorithm always converges monotonically. And it will find the optima when updating is small. Note this might not be the global optima. 


## Mixture of Gaussian

I am still working on this subject. 

