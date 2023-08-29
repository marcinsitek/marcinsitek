import requests
import json
from typing import List
from dataclasses import replace

from cpi_prediction.objects.series import Series

class FredAPI:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        session: requests.Session = None,
    ):
        self._url = api_url
        self._key = api_key
        self._session = session if session is not None else requests.Session()

    def _process_response(self, response: dict) -> List[dict]:
        data = []
        for observation in response["observations"]:
            row = {
                "date": observation["date"],
                "value": None if observation["value"] == "." else observation["value"],
            }
            data.append(row)
        return data

    def get_series(self, series: Series) -> Series:
        url = (
            self._url
            + "series_id={0}&api_key={1}&file_type={2}&observation_start={3}&observation_end={4}".format(
                series.id, self._key, "json", series.start_dt, series.end_dt
            )
        )
        resp = self._session.get(url)
        data = self._process_response(json.loads(resp.text))
        field = {'data': data}
        return replace(series, **field)
