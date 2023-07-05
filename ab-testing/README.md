# ab-testing

## Introduction
---
The goal of this project is to showcase a demo implementation of methods used for validating results of A/B tests. 
The project is outlined as follows: 
* `*.ipynb` files demonstrate various methods
* `data_generator.py` contains the `DataGenerator()` class

## Data generating process
---
Main assumptions:
* context: mobile game players making in-app purchases
* players:  $i=1, ..., N$ where $N = 10000$
* test variants:    *A*,*B* where $P(variant_{i}=A)=p$, $P(variant_{i}=B)=1-p$ and $p=0.5$ 
* payers:   $payer_{i}\sim Bernoulli(p)$ where $p=0.1$
* sum of payments: $payment\_sum_{i}\sim Gamma(k, \theta)$ where $k=0.5$, $\theta=2.0$
* constant scale for payments: $C=100$
* shift for variant *B*: $\mu = 1.3$
* see details in `data_generator.py`

## Methods
---
| File              | Method                          | Reference        |
| ----------------- | ------------------------------- | ------------------ |
| `classical.ipynb` | Two-sample test for means       | [1]|
| `classical.ipynb` | Two-sample test for proportions | [1]|
| `classical.ipynb` | Chi-square test for homogeneity | [2]|
| `bootstrap.ipynb` | Nonparametric bootstrap test for means | [3]|
| `bootstrap.ipynb` | Studentized nonparametric bootstrap test for means | [3]|
| `cuped.ipynb`     | CUPED                           | [4]|

## Reference
---
* [1] Janina Jóźwiak, Jarosław Podgórski, 'Statystyka od podstaw' (2022) 
* [2] 'Pearson's chi-squared test' (2003) Wikipedia. Available [here](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test)
* [3] Davison, A.C. and Hinkley, D.V. (1997) Bootstrap Methods and Their Application, Cambridge University Press.
* [4] A Deng, Y Xu, R Kohavi, T Walker, 'Improving the sensitivity of online controlled experiments by utilizing pre-experiment data' (2013)