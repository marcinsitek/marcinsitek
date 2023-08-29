import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from cpi_prediction.config import SeriesId

@dataclass
class Series:
    id: SeriesId
    start_dt: datetime.datetime
    end_dt: datetime.datetime
    data: Optional[List[dict]] = None
