import click
import pandas as pd
import numpy as np
import mlflow

from datetime import datetime

from cpi_prediction.data.data_retriever import DataRetriever
from cpi_prediction.data.data_preprocessor import DataPreprocessor
from cpi_prediction.data.data_uploader import DataUploader
from cpi_prediction.data.data_utils import get_max_date, truncate_table
from cpi_prediction.model.model import ARIMAModel
from cpi_prediction.model.params import P, D, Q
from cpi_prediction.objects.series import Series
from cpi_prediction.api.fred import FredAPI
from cpi_prediction.detrender import Detrender
from cpi_prediction.config import (
    DB, 
    TABLE,
    Y, 
    EXOG, 
    START_DATE, 
    END_DATE, 
    API_ENDPOINT, 
    API_KEY, 
    MLFLOW_EXPERIMENT,
    SeriesId,
    logger
)


def update():
    max_date = get_max_date(DB, TABLE, 'date')
    logger.info(f"Updating started:    max_date={max_date}")
    if (END_DATE - datetime.fromisoformat(max_date).date()).days > 60:
        start_date = START_DATE.strftime("%Y-%m-%d")
        end_date = END_DATE.strftime("%Y-%m-%d")
        fred = FredAPI(API_ENDPOINT, API_KEY)
        dfs = []
        for s in SeriesId:
            logger.info(f"     Retrieving started:    {s.value}")
            series = Series(id=s.value, start_dt=start_date, end_dt=end_date)
            series_with_data = fred.get_series(series)
            df = pd.DataFrame(series_with_data.data)
            df['series'] = series_with_data.id
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        logger.info(f"     Truncating started")
        truncate_table(DB, TABLE)
        du = DataUploader(DB, TABLE)
        du.upload(df)
    new_max_date = get_max_date(DB, TABLE, 'date')
    logger.info(f"Updating completed:    max_date={new_max_date}")



def train(date_from: datetime = None, date_to: datetime = None) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{DB}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    if date_from is None:
        date_from = START_DATE
    if date_to is None:
        date_to = END_DATE

    with mlflow.start_run():
        logger.info(f"Training started:    date_from={date_from}   date_to={date_to}")
        mlflow.set_tag("run_id", mlflow.active_run().info.run_id)
        mlflow.set_tag("date_from", date_from)
        mlflow.set_tag("date_to", date_to)

        dr = DataRetriever(
            db=DB,
            start_dt=date_from,
            end_dt=date_to
        )
        data = dr.retrieve()
        dp = DataPreprocessor(data)
        processed_df = dp.preprocess()
        y = processed_df.loc[:, Y]
        dt = Detrender()
        dt.fit(y)
        model = ARIMAModel(p=P, d=D, q=Q)
        mlflow.log_param("p", P)
        mlflow.log_param("d", D)
        mlflow.log_param("q", Q)
        logger.info(f"     Fitting started:    p={P} d={D} q={Q}")
        model.fit(
            y=dt.transform(y), 
            exog=processed_df.loc[:, [f"l_d_ln_{x}" for x in EXOG]]
        )
        aic = model.get_aic()
        rmse = (np.mean((model.get_residuals())**2))**(1/2)
        logger.info(f"     Fitting completed:    AIC={round(aic,2)} train RMSE={round(rmse,2)}")
        mlflow.log_metric("aic", aic)
        mlflow.log_metric("rmse", rmse)
        exog = processed_df.loc[:, [f"d_ln_{x}" for x in EXOG]].values[-1]
        t = len(y)
        predicted_value = (
            model.predict(exog=exog) 
            + (dt.coef_[0] + dt.coef_[1]*(t+1) + dt.coef_[2]*(t+1)**2)
        )
        last_date = processed_df.index[-1]
        next_date = last_date + 1
        logger.info(f"Training completed")
    return {next_date.strftime('%Y-%m'):round(predicted_value,2)}



@click.group()
def cpi():
    ...

@cpi.command("train")
@click.option("--date_from", type=click.DateTime(formats=["%Y-%m-%d"]), required=False)
@click.option("--date_to", type=click.DateTime(formats=["%Y-%m-%d"]), required=False)
def train_cmd(**kwargs):
    prediction = train(**kwargs)
    print(prediction)


@cpi.command("update_and_train")
@click.option("--date_from", type=click.DateTime(formats=["%Y-%m-%d"]), required=False)
@click.option("--date_to", type=click.DateTime(formats=["%Y-%m-%d"]), required=False)
def update_and_train_cmd(**kwargs):
    update()
    prediction = train(**kwargs)
    print(prediction)


if __name__ == "__main__":
    cpi()
