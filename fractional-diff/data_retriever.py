import numpy as np
import pandas as pd
import sqlite3
import datetime


class DataRetriever:
    QUERY = """
        select 
            *, 
            substr(date,1,4) || '-' || substr(date,5,2) || '-' || substr(date,7,2) as dt 
        from cryptocurrencies
        where dt >= ? and dt <= ?;
    """

    def __init__(
        self, db: str, start_dt: datetime.date = None, end_dt: datetime.date = None
    ) -> None:
        self.db: str = db
        self.start_dt: datetime.date = start_dt
        self.end_dt: datetime.date = end_dt
        self.data: pd.DataFrame = None

    def _retrieve_data_from_db(self) -> pd.DataFrame:
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        params = (self.start_dt.strftime("%Y-%m-%d"), self.end_dt.strftime("%Y-%m-%d"))
        res = cur.execute(DataRetriever.QUERY, params)
        data = res.fetchall()
        columns = [c[0] for c in cur.description]
        data = pd.DataFrame(data, columns=columns).fillna(np.nan)
        return data

    def retrieve(self) -> pd.DataFrame:
        self.data = self._retrieve_data_from_db()
        assert self.data is not None
        return self.data
