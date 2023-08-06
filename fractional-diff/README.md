# fractional-diff

## Introduction

The goal of this project is to showcase a demo implementation of fractional differentiation for time series based on [1].
The project is outlined as follows: 
* `demo.ipynb` file demonstrates the logic of the approach
* `*.py` files contain source code
* `data.db` file is an SQLite database where data is stored

## Business Context

The goal of fractional differentation is to make a time series stationary while preserving at the same time as much memory as possible. 

## Data 

Key information:
* Source: stooq.pl
* Daily OHLC data  $t=1, ..., N$ where $N = 910$
* Data range: *2021-01-01* - *2023-06-29*
* Ticker: ETH (Ethereum)
* $y_{t}$: Close price, PLN
* Inserted into an SQLite database: *data.db*

## Method

Let's consider a time series:
$$y_{t} = \{y_{1}, y_{2}, ..., y_{\tau}\}$$
and let's define a lag operator *L* such that $L^{k}y_{t}=y_{t-k}$
for $k\geq0$ and $t>0$. Then we can express the first-order differentiation in the following way:
$$\Delta y = y_{t}-y_{t-1} = y_{t}-Ly_{t} = (1-L)y_{t}$$
And more generally for any positive integer *d* we can define differentation of order *d*:
$$\Delta^{d} y = (1-L)^{d}y_{t}$$
Fractional differentiation allows *d* to be a real number such that $d\in[0,1]$ and hence $(1-L)^{d}$ can be expanded to a series of weights using the binomial expansion, see [1].
The function for calculating these weights is implemented in `FractionalDifferentiation()`

A quasi-optimal *d* is found over a discreet space `np.linspace(0, 1, 20)` such that it maximizes the Pearson's correlation coefficient between $y_{t}$ and $\Delta^{d} y$ and minimizes the augmented Dickey–Fuller test statistic s.t it is greater than the critical value of the test for the 5% confidence interval.


## Reference

* [1] Janusz Gajda & Rafał Walasek, 2020. "Fractional differentiation and its use in machine learning," Working Papers 2020-32, Faculty of Economic Sciences, University of Warsaw.

