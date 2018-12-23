---
layout: single
mathjax: true
toc: true
toc_sticky: true
category: Machine Learning
tags: [notes]
qr: machine_learning_notes.png
title: Variational Inference
share: true
permalink: /MachineLearning/bayes_vi/
sidebar:
  nav: "MachineLearning"
---

# Review of EM algorithm

In [last section](https://wei2624.github.io/MachineLearning/usv_em/), we talked about how EM algorithm works and why it works in general. The idea is to find a proper latent varibale  which can be integrated out to get back to joint distribution. The latent variable comes in to facilitate the math challenges when we are trying to find the optimal parameters for joint distributions. With the latent variable, we can find an point estimate of parameters which result in the highest joint likelihood. 

However, EM does not always work. When EM does not work, VI comes into play to solve the issue. We will see some exapmples where EM does not work and build the transition from EM to VI. 

# Model Setup

In this section, we will see different model setups and how EM fails in which stage. 

## Model V1

In this version of setup, we have the simplest model setup. Formally, we have:

$$y_i \sim \mathcal{N}(x_i^Tw,\alpha), w\sim\mathcal{N}(0,\lambda I)$$

We assume each sample is iid and dimensions are matched. There is no prior distribution on x. Thus, we are in discriminative setting not generative setting where we have the prior model on x. 

We can calculate the posterior on w using Bayes rule:

$$p(w\lvert x,y)=\frac{p(y\lvert w,x)p(w)}{p(y\lvert x)} = \mathcal{N}(\mu,\Sigma)$$

This does not require any iterative algrotihm and is very easy to calculate in the closed form. However, we notice that there are two parameters that we need to pre-define: $\alpha,\lambda$. These parameters can affect the final performance. In general, there are two ways to work on it:

1 We can use cross-validation on these two parameters and select the one witht the best performance. 

2 In Bayesian, we can put a prior distribution on the parameters and learn it!

## Model V2

So we want to pick up a conjugate gamma propr on $\alpha$. Then, we have:

$$y_i \sim \mathcal{N}(x_i^Tw,\alpha), w\sim\mathcal{N}(0,\lambda I), \alpha\sim Gamma(a,b)$$

In this case, $alpha$ becomes another model variable that we want to learn like w. We can do one the following:

1 Try to learn the full posterior on $\alpha$ and w (We cannot assume w and $\alpha$ are independent). 

2 Learn a point estimate of w and $\alpha$ based on MAP inference. 

3 Integrate out some variables to learn the others. 

Let's look at each one:

1 The full posterior of $\alpha$ and w is:

$$ p(w,\alpha\lvert y,x) = \frac{p(y\lvert w, \alpha, x)p(w)p(\alpha)}{\int\int p(y\lvert w, \alpha, x)p(w)p(\alpha) dwd\alpha}$$

We realize the normalizing constant cannot be caluclated in the closed form. We can use other approximation method to approximate it such as Laplace or MCMC sampling. 

2 We can do MAP over w and $\alpha$:

$$w,\alpha = \arg\max_{w,\alpha} \ln  p(y,w,\alpha\lvert x)$$

$$=\arg\max_{w,\alpha} \ln p(y\lvert w,\alpha,x) + \ln p(w) + \ln p(\alpha)$$

We can use coordinate descent algorithm to optimize each parameter. However, this will not tell us about uncertainty. Rather, it will only tell us an point estimate of the model parameters. 

3 We can do marginal likelihood as:

$$p(y\lvert w x) = \int p(y,w, \alpha\lvert x) d\alpha$$

With this marginal likelihood, we can:

(1) try to find posterior inference for $p(w\lvert y,x)\propto p(y\lvert w,x)p(w)$. However, this will not work out since $p(y\lvert w,x)$ is resulted from the integral of $\alpha$, which is not a Gaussian anymore. Rather, it is student-t distribution. This is not conjugate with prior anymore. So it does not work. 

(2) Another option is to maximize $p(y,w\lvert x)$ over w using MAP using gradient method. This is where EM could possibly come in. 

## EM for Model V2

So we want to treat $\alpha$ as the latent variable. We are trying to find a point estimate of w to maximize the marginal distribution $p(y,w\lvert x)=\int p(y,w,\alpha\lvert x)\alpha$. 

Based on the above discussion, we can write EM master equation:

$$\ln p(y,w\lvert x)=\underbrace{\int q(\alpha)\ln\frac{p(y,w,\alpha\lvert x)}{q(\alpha)}d\alpha}_{\mathcal{L}(w)} + \underbrace{\int q(\alpha)\ln\frac{q(\alpha)}{p(\alpha\lvert y,w,x)}d\alpha}_{KL(p\lvert\lvert q)}$$

E-step:

$$\begin{align}
p(\alpha\lvert y,w,x) &\propto \prod_{i=1}^N p(y_i\lvert \alpha.w,x_i)p(\alpha) \\
&\propto \alpha^{\frac{N}{2}}\exp(-\frac{\alpha}{2}\sum\limits_{i=1}^N (y_i-x_i^Tw)^2)\times \alpha^{a-1}\exp(-b\alpha) \\
&=Gamma(a+\frac{N}{2},b+\frac{1}{2}\sum\limits_{i=1}^N (y_i-x_i^Tw)^2)
\end{align}$$ 

Now, we should set $q_t(\alpha) = p(\alpha\lvert y,w_{t-1},x)$, and then calculate the loss function:

$$\begin{align}
\mathcal{L}(w) &= \mathbb{E}_q [\ln p(y,\alpha\lvert w,x)p(w)] + \mathbb{E}_q [\ln q(\alpha)] \\
&= -\frac{\mathbb{E}_{q_t} [\alpha]}{2} \sum\limits_{i=1}^N (y_i -x_i^Tw)^2 - \frac{\lambda}{2}w^Tw + \text{const. w.r.t. w}
\end{align}$$

M-step:

Then, we have to make sure that we can maximize the loss function in a closed form. Otherwise, there is no point of doing this. We can try and find (I will give the math later):

$$w_t = (\lambda I + \mathbb{E}_{q_t} [\alpha]\sum\limits_{i=1}^N x_i x_i^T)^{-1}(\sum\limits_{i=1}^N \mathbb{E}_{q_t}[\alpha]y_i x_i)$$

We can plug $\mathbb{E}_{q_t}[\alpha]$ into the above so that we get the updating form of w. 


Note that there are a few things that are different than we did EM last time.

(1) In the previous case, we introduced a latent variable to reduce the math complexity. However, in this case, the latent variable $\alpha$ has the interpretation, which is related to the observation noise. So this latent varibale has its own meaning. 

(2) In this case, we have w and $\alpha$ to learn. However, we make a compromise by learning a point estimate and a conditional posterior of $\alpha$. It is hard for us to learn point estimates of two variables. Surly, you could have done the reverse. That is, you can learn a point estimate of $\alpha$ and a conditional posterior of w. 


## Model V3

The EM algorithm works in model version 2. Let's see a new version of model, namely version 3. In this setup, we have one more variable to work with. 

$$ y_i \sim \mathcal{N}(x_i^Tw, \alpha^{-1}), w\sim \mathcal{N}(0,\lambda I),\alpha\sim Gamma(a,b),\lambda\sim Gamma(e,f)$$

Then, the marginal distribution is:

$$p(y,w\lvert x) = \int\int p(y,w,\alpha,\lambda\lvert x)d\alpha d\lambda$$

## EM for Model V3

Note that we have two latent variables now instead of one in V2. The EM master equation is:

$$\begin{align}
\ln p(y,w\lvert x) &= \int\int q(\alpha,\lambda)\ln\frac{p(y,w,\alpha,\lambda\lvert x)}{q(\alpha,\lambda)}d\alpha d\lambda + \\
& \int\int q(\alpha,\lambda)\ln\frac{q(\alpha,\lambda)}{p(\alpha,\lambda\lvert y,w,x)}
\end{align}$$

Note that $\ln p(y,w\lvert x)$ in this model setup is different from the one in V2, although the form is the same. That is, the underlying distribution is essentially different. 

E-step:

As usual, we have:

$$\begin{align}
p(\alpha,\lambda\lvert y,w,x) &= \frac{p(y\lvert w,\alpha,x)p(\alpha)p(w\lvert\lambda)p(\lambda)}{\int\int p(y\lvert w,\alpha,x)p(\alpha)p(w\lvert\lambda)p(\lambda) d\lambda d\alpha} \\
&= \underbrace{\frac{p(y,\alpha\lvert w,x)}{\int p(y,\alpha\lvert w,x)d\alpha}}_{p(\alpha\lvert y,w,x)}\underbrace{\frac{p(w,\lambda)}{\int p(w,\lambda)d\lambda}}_{p(\lambda\lvert w)}
\end{align}$$

We can calculate the conditional posterior of both parameters:

$$p(\alpha\lvert,y,w,x) = Gamma(a+\frac{N}{2},b+\frac{1}{2}\sum\limits_{i=1}^N (y_i-x_i^Tw)^2)$$

$$p(\lambda\lvert w) = Gamma(e+\frac{d}{2},f+\frac{1}{2}w^Tw)$$

M-step:

We have the same form for the updating as:

$$w_t = (\mathbb{E}_{q_t}[\lambda] I + \mathbb{E}_{q_t} [\alpha]\sum\limits_{i=1}^N x_i x_i^T)^{-1}(\sum\limits_{i=1}^N \mathbb{E}_{q_t}[\alpha]y_i x_i)$$

The only difference is that we have the expectation for $\lambda$ as well. This worked out since we can factorize the full conditional posteriors of the two variables as:

$$q(\alpha,\lambda) = p(\alpha,\lambda\lvert y,w,x) = p(\alpha\lvert y,w,x)p(\lambda\lvert w) = q(\alpha)q(\lambda)$$

This is not always true. In this case, we just make it factorizeable. Eventually, we will have a point estimate and full posterior of variables. 

The question is that what if we cannot factorize or find the posterior of the variables. In this case, we have the problem. If we cannot calculate or factorize the posterior, we cannot move on with the EM algorithm. This is where VI might come in. 

# From EM to VI

In EM, we can learn posteriors of all variables except for the one that is learned as point estimate. So let's make it learn all variables in posteriors. For model V3, we can have:

$$p(y\lvert x) = \int\int\int p(y,w,\alpha,\lambda) d\alpha d\lambda dw$$

Note that different from EM, there is nothing to optimize on the left hand side. We can further write:

$$\begin{align}
\ln p(y\lvert x) &= \int\int\int q(\alpha,\lambda,w)\ln \frac{p(y,w,\alpha,\lambda\lvert x)}{q(\alpha,\lambda,w)}d \alpha d\lambda dw + \\
& \int\int\int q(\alpha,\lambda,w)\ln \frac{q(\alpha,\lambda,w)}{p(\alpha,\lambda,w\lvert y,x)}d \alpha d\lambda dw
\end{align}$$

Again, there is nothing to optimize on the left hand side. In addition, we cannot complete our E-step since we cannot calculate the full posterior:

$$q(\alpha,\lambda, w) = p(\alpha,\lambda, w\lvert y,x)$$

So let's look at the VI equation:

$$\begin{align}
\ln p(y\lvert x) &= \int\int\int q(\alpha,\lambda,w)\ln \frac{p(y,w,\alpha,\lambda\lvert x)}{q(\alpha,\lambda,w)}d \alpha d\lambda dw + \\
& \int\int\int q(\alpha,\lambda,w)\ln \frac{q(\alpha,\lambda,w)}{p(\alpha,\lambda,w\lvert y,x)}d \alpha d\lambda dw
\end{align}$$

In EM, we want to have a proper q distribution to find out an point estimate of w. Both left and right hand side of EM equation are optimized. In VI, there is nothing to optimize on the left hand side. Thus, in VI, we view this equation differently. In particular, we are more interested in q distributions and how to pick them up. 

