# mmm-bayesian

## Introduction
---
The goal of this project is to showcase a demo implementation of a media-mix model using a Bayesian approach. 
The project is outlined as follows: 
* `demo.ipynb` file demonstrates the logic of the approach
* `*.py` files contain source code
* `data.db` file is an SQLite database where data is stored

Caveats:
* The model is not ready for production environment
* Further analysis and tuning is required

## Business Context
---
Media mix modeling (MMM) is an analytical method used to measure the impact of marketing activities on a KPI e.g. conversions, revenue. It can be used to optimize advertising mix in the absence of deterministic attribution. More information can be found in [1]. From the statistical standpoint it is a regression problem.

## Data 
---
Key information:
* Dataset provided as part of the Robyn package [2]
* Weekly data  $t=1, ..., N$ where $N = 208$
* Data range: *2015-11-23* - *2019-11-11*
* Media spend for the following marketing channels: *tv, ooh, print, search, facebook*
* Additional marketing channels: *newsletter, events*
* Inserted into an SQLite database: *data.db*

## Model
---
Model specification follows [3] and [4] i.e. the target $y_{t}$ variable is modelled by the following equation:
$$y_{t} = \tau + \sum_{m=1}^{M}\beta_{m}x_{t,m}^{*}+\sum_{c=1}^{C}\gamma_{c}z_{t,c}+\epsilon_{t}$$
where:
* $y_t$ is the target variable i.e. revenues
* $x_{t,m}^{*}=saturation(adstock(x_{t,m},\cdot), \cdot)$ , $c=1,...,C$ are transformed media variables
* *adstock()* is implemented in a simpler way than it is in [3] i.e. $adstock(x_{t},\theta)=x_{t}+\theta*x_{t-1}$, see `model.py` and `adstock.py`
* *saturation()* is modelled using the Hill function i.e. $hill(x_{t}; k,n)=1 / (1 + (x / k)^{n})$, see `model.py` and `saturation.py`
* $z_{t,c}$ , $c=1,...,C$ are control variables: events(binary) and competitor's sales (numeric)
* $\tau$ is an intercept which can be interpreted as base revenues
* $\epsilon_{t}$ is white noise with standard assumptions for a linear model
* Seasonality is not addressed

## Fitting
---
The optimal parameters $\alpha$, $\theta$, $a_{i}$ and $b_{i}$ for $i = 1, .., M$ are found using methods for Bayesian modelling from `PyMC3`, see `model.py`

The assumed a priori parameter distributions are the following:
* $\beta_{m} \sim Beta(2,2)$
* $\gamma_{c} \sim Beta(2,2)$
* $\theta_{m} \sim Beta(2,2)$
* $k_{m} \sim Gamma(3)$
* $n_{m} \sim Beta(2,2)$

Variables are scaled using `MinMaxScaler()` from `scikit-learn`.

## Performance
---
Model performance is assessed using:
* a hold-out test set corresponding to the last 20% of observations: `train_test_split(..., shuffle=False, test_size=0.2, ...)`
* 4 performance metrics: $R^{2}$, *RMSE*, *NRMSE*, *MAPE*, *SMAPE* which are implemented in `metrics.py`

## Reference
---
* [1] 'Media mix modeling (MMM)' (2023) AppsFlyer . Available [here](https://www.appsflyer.com/glossary/media-mix-modeling/)
* [2] Bernardo Lares (2023), 'Robyn'. More information [here](https://github.com/facebookexperimental/Robyn) and [here](https://cran.rstudio.com/web/packages/Robyn/Robyn.pdf)
* [3] Yueqing Wang, Yuxue Jin, Yunting Sun, David Chan, Jim Koehler (2017), 'A Hierarchical Bayesian Approach to Improve Media Mix Models
Using Category Data', Google Inc.
* [4] Yuxue Jin, Yueqing Wang, Yunting Sun, David Chan, Jim Koehler (2017), 'Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects', Google Inc.

