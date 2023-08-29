import os
import logging
from enum import Enum
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger('cpi-prediction')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(message)s')

class SeriesId(Enum):
    CPIAUCSL = "CPIAUCSL"
    FEDFUNDS = "FEDFUNDS"
    MCOILWTICO = "MCOILWTICO"

# Dates
START_DATE = datetime.fromisoformat('1999-01-01').date()
END_DATE = date.today()
# API
API_ENDPOINT = 'https://api.stlouisfed.org/fred/series/observations?'
API_KEY = os.environ.get('API_KEY')
# DB
ROOT_DIR = Path(__file__).parent.resolve()
DB = ROOT_DIR / 'data' / 'data.db'
TABLE = 'fred'
# Y, X
Y = SeriesId.CPIAUCSL.value
EXOG = [SeriesId.FEDFUNDS.value, SeriesId.MCOILWTICO.value]
# mlflow
MLFLOW_EXPERIMENT = "cpi-prediction"
