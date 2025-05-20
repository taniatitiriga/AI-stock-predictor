
from datetime import datetime, timedelta

# configuration
TICKER = 'AAPL' # default
TODAY_STR = datetime.today().strftime('%Y-%m-%d')
YESTERDAY_STR = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

YEARS_FOR_BENCHMARK = 3
BENCHMARK_END_DATE = YESTERDAY_STR
BENCHMARK_START_DATE = (datetime.today() - timedelta(days=365 * YEARS_FOR_BENCHMARK)).strftime('%Y-%m-%d')
START_DATE = (datetime.today() - timedelta(days=365 * 4)).strftime('%Y-%m-%d')
END_DATE = TODAY_STR 

LOOK_BACK_WINDOW = 60
TRAIN_SPLIT_RATIO = 0.8
EPOCHS = 15
BATCH_SIZE = 32

# multi-feature input
FEATURES_TO_USE = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'rsi_14',
    'macd_12_26_9', 'macdh_12_26_9', 'macds_12_26_9'
]
TARGET_COLUMN = 'close' # predicted