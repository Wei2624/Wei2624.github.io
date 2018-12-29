---
layout: single
mathjax: true
category: Stats
tags: [notes,math,interview]
qr: interesting_probability_questions.png
title: Interesting Probability Questions
permalink: /posts/probability-questions/
---

In this blog, I will keep updating interesting probability questions as time goes by. The questions listed are likely to apprear in interview. The answers written by me will be given as well. Please be skeptical about the answers. 

**Note: You can share this post to social network at the bottom.**


**1 In coin tossing, assuming it is a fair coin, what is the expected tosses of the event that a head comes before a tail?**

Solution: Let A denote the event that a head comes before a tail and $H_i$ denote the event that first toss is a head. Thus, we have: 

$$E(A) = 0.5\times E(A | H_1) + 0.5\times E(A | T_1) \label{1}\tag{1}$$

To solve this, we look into each term. 

$$E(A | H_1) = 0.5\times 2 + 0.5\times (1 + E(A | H_1) \label{2}\tag{2}$$

$$E(A | T_1) = 1 + E(A) \label{3}\tag{3}$$

From \ref{2}, we have $E(A \lvert H_1) = 3$. We plug this with \ref{3} to \ref{1} so as to solve it: $E(A) = 4$

**2 In a similar manner, we have a fair coin, what is the expected tosses of the event that two heads comes consecutively?**

Solution: Test it with yourself. My answer is 6. 

**3 Given that we have a fair coin, what is the expected tosses of the event that two heads appear?**

Solution: This question requires a bit tricks. Intuitively, suppose that we have probability p to have a head, if we toss n times, the expected times of heads is $np$. If we set it to k, the number of heads we want, then $n=\frac{k}{p}$. So the quick answer is 4. 

Now, I show the completed steps. Let $X = X_1 + X_2 + \dots + X_k$ where k is the number of heads we want (k=2 in this case) and $X_i$ denotes the number of tosses of getting ith head after getting (i-1) heads. Due to the linearity of expectation, we have $E(X) = E(X_1) + E(X_2) + \dots + E(X_k)$. Since each individual $X_i$ is identically distributed and a geometric random variable, $E(X_i) = \frac{1}{p}$. Then, it is easy to have $E(X) = \frac{k}{p}$.

**4 Tossing a fiar coin, what is the probability of the event that HTT comes before HHT?**

Solution: This quesiton is a bit trickier than the Q1 since it asks for a probability instead of a expected number. However, the idea is essentially the same. Let A denote the event that HTT comes before HHT. Then, it is easy to say:

$$P(A) = P(A \lvert H_1)P(H_1) + P(A \lvert T_1)P(T_1) = 0.5P(A \lvert H_1) + 0.5P(A \lvert T_1) \label{4}\tag{4}$$

$$P(A \lvert H_1) = P(A \lvert H_1 H_2)P(H_2) + P(A \lvert H_1 T_2)P(T_2) = 0 + 0.5P(A \lvert H_1 T_2) \label{5}\tag{5}$$

$$P(A \lvert H_1 T_2) =(A \lvert H_1 T_2 H_3)P(H_3) + P(A \lvert H_1 T_2 T_3)P(T_3) = 0.5P(A \lvert H_1) + 0.5 \label{6}\tag{6}$$

Combining \ref{5} and \ref{6}, we have $P(A\lvert H_1) = \frac{1}{3}$. From \ref{4}, we have $P(A) = P(A\lvert H_1) = \frac{1}{3}$.

**5 Interview n candidates for assistant in order, hire candidate i if better than the current assistant,how many times of hiring a new assistant?**

Solution: Let $X$ denote # times of hiring a new assistant and $Z_i = \mathbb{1}$[hire i-th candidate]. The probablity of hiring i-th candidate is $\frac{1}{i}$. Thus, 

$$E[X] = E[\sum\limits_{i=1}^{n} Z_i] = \ln{n}$$

**6 There are n cards facing down. For each card, guess randomly and turn it over for checking. Get one point for a correct guess. What is expected number of points?**

Solution: Let $Y_i = \mathbb{1}$[i-th guess correct] and $X = \sum\limits_{i=1}^{n} Y_i$. The probability of guessing right is $\frac{1}{n}$ Thus,

$$E[X] = \sum\limits_{i=1}^{n} E[Y_i] = 1$$

**7 How many people do you need until you expect to find two people with the same birthday?**

Solution: Let $X_{ij} = \mathbb{1}$[persons i and j have the same birthday], then $E[X_{ij}] = \frac{n}{n^2}$. The total expected number of pairs of same birthday is $E[\sum X_{ij}]=\sum E[X_{ij}]=\frac{k(k-1)}{2n}$ where $k$ is the number of people we should have and $n$ is 365 days in this case. Let the equation equal to 1 so that we can solve for $k$.


**8 You sample 36 apples from a farm's harvest of over 20000 apples. The mean weight of the samles is 112 grams with 40 gram STD. What is the probabiliyt that the mean wieght os all 20000 apples is within 100 to 124 grams?**

Solution: This question involves the knowledge of mean and STD of sample mean and can be extended to Central Limit Theorem. The questions actually asks to find the distribution of sample mean. What is the mean and STD of sample mean? Thus, let $\mu$ be the mean weights of all apples, $\sigma$ be the standard deviation of all apples, $\bar{x}$ be the sample mean, S be the sample STD, $\mu_{\bar{x}}$ be the mean of sample mean, $\sigma_{\bar{x}}$ be the STD of sample mean and $X_i$ be the weight of an apple sampled at i-th time where $1\leq i \leq n$ and n the number of samples. The last two are of the interests. 

$$\mu_{\bar{x}} =  \text{mean}(\frac{\sum\limits_{i=1}^{n} X_i}{n}) = \frac{\text{mean}(\sum\limits_{i=1}^{n} X_i)}{n} = \frac{\sum\limits_{i=1}^{n} \text{mean}(X_i)}{n} = \bar{x}$$

$$\sigma_{\bar{x}} = \text{Var}(\frac{\sum\limits_{i=1}^{n} X_i}{n}) = \frac{\text{Var}(\sum\limits_{i=1}^{n} X_i)}{n^2} = \frac{\sum\limits_{i=1}^{n} \text{Var}(X_i)}{n^2} = \frac{S}{\sqrt{n}} \text{where} S \approx \sigma$$

Extension: Central Limit Theorem.

