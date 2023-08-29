import pandas as pd
import sqlite3

from typing import List

class DataUploader:
    def __init__(self, db: str, table: str) -> None:
        self.db: str = db
        self.table: str = table

    def _rows_to_insert(self, data: pd.DataFrame) -> List[tuple]:
        return list(data.itertuples(index=False, name=None))

    def upload(self, data: pd.DataFrame) -> None:
        rows = self._rows_to_insert(data)
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.executemany(f"INSERT INTO {self.table} VALUES(?, ?, ?)", rows)
        con.commit() 

