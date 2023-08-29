# cpi-prediction

## Introduction
Package can be used to predict the value of the Consumer Price Index for the next month.
The project is outlined as follows: 
* `./notebooks/demo.ipynb` file demonstrating the model building process
* `./src/cpia_prediction/*` files and directories containing source code
* `./src/cpia_prediction/data/data.db` file with an SQLite database where data is stored

## Instalation
1. Install in editable mode:
```
pip install -e .
```
2. Set an enviroment variable to store your api key:
```
export API_KEY=XXX
```
3. Set an environment variable with the path do `data.db` in order to store tracking data:
```
export MLFLOW_TRACKING_URI=sqlite:////path/to/data.db
```

## Run
1. Run mlflow server
```
mlflow server \
    --backend-store-uri sqlite:////path/to/data.db \
    --default-artifact-root /path/to/mlruns \
    --host 0.0.0.0
    --port 5000
```
2. Run the following command to predict value for the next period given data in `data.db`
```
mlflow run . \
    --entry-point train \
    --experiment-name="cpi-prediction" \
    -P date_from="1999-01-01" \
    -P date_to="2023-07-01"
```
3. Run the following command to retrieve data from FRED API, save it in `data.db` and then make a prediction:
```
mlflow run . \
    --entry-point update_and_train \
    --experiment-name="cpi-prediction" \
    -P date_from="1999-01-01" \
    -P date_to="2023-07-01"
```

## Data 
Key information:
* Data retrieved from FRED, Federal Reserve Bank of St. Louis
* Data retrieved through an API, see `src/api/fred.py` and `src/config.py`
* Monthly data  $t=1, ..., N$ where $N = 293$
* Data range: *1999-01-01* - *2023-07-01*
* Time series being predicted $y_{t}$ : *CPIAUCSL* [1]
* Exogenous variables: *FEDFUNDS*, *MCOILWTICO*
* Inserted into an SQLite database: *data.db*

## Model
An autoregressive integrated moving average model with exogenous variables *ARIMAX(p,d,q)* as presented in [2] fitted to $y_{t}$ using the Box-Jenkins approach:
$$(1 - L)^{d}(y_{t} - m_{t}) = \sum_{j=1}^{p}\alpha_{j}y_{t-j} + \sum_{j=0}^{q}\theta_{j}\epsilon_{t-j} + \sum_{i=1}^{2}\beta_{i}(1 - L)x_{i,t-1}$$
where:
* $\theta_{0}=1,   \epsilon_{t} \sim WN(0, \sigma^{2})$
* $L$ is the lag operator defined as $L^{d}y_{t} = y_{t-d}$
* $y_t$ is the target time series i.e. CPIAUCSL
* $m_{t}$ in the deterministic mean, modeled as $\delta_{0} + \delta_{1}t + \delta_{2}t^{2}$, see `demo.ipynb` to see that all $\delta_{i}, i=0,1,2$ are statistically significant
* $x_{i, t-1}$ is one of the two exogenous variables, they both prove to be difference-stationary
* Seasonality is not addressed in this demo example 

## Finding p, d, q
The optimal parameters $p$, $d$ and $q$ are found using a two-step approach:
1. Possible values of $d$ are determined using an Augmented Dickey-Fuller test for stationarity from `statsmodels`
```
adfuller()
```
1. Possible values of $p$ and $q$ are determined by analyzing correlograms with autocorrelation and partial autocorrelation functions from `statsmodels`
```
sm.graphics.tsa.plot_acf()
sm.graphics.tsa.plot_pacf()
```
2. Optimal values are found using time-series cross validation, see `/notebooks/demo.ipynb`


The metrics being optimized during cross validation are: 
* *MAIC* mean Akaike's criterion and 
* *RMSE* calculated for out-of-sample test value for each of $K = 40$ folds.

A parsimonious model is chosen with both AR and MA parts which yields the lowest value of cross-validated RMSE.

The chosen parameters *p, d, q* can be found in `model/params.py`

## Final checks
Based on appriopriate test statistics the final model has residuals which:
* are normally distributed: `Jarque-Bera (JB):	149.85`
* exibit no correlation of order 1: `Ljung-Box (L1) (Q):	0.20`
* have some form of heteroskedasticity: `Heteroskedasticity (H):	1.59`

All coefficients except one are statistically significant at the level of 1%. 

The root of the AR part satisfies the stability condition: $|0.4530| < 1$


## Reference
* [1] U.S. Bureau of Labor Statistics, Consumer Price Index for All Urban Consumers: All Items in U.S. City Average [CPIAUCSL], retrieved from FRED, Federal Reserve Bank of St. Louis. Available [here](https://fred.stlouisfed.org/series/CPIAUCSL), August 2023
* [2] Autoregressive–moving-average model, Wikipedia. Available [here](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model#ARMAX)


