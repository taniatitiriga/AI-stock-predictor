# configuration
TICKER = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'
LOOK_BACK_WINDOW = 60
TRAIN_SPLIT_RATIO = 0.8
EPOCHS = 25
BATCH_SIZE = 32

# multi-feature input
FEATURES_TO_USE = [
    'open',
    'high',
    'low',
    'close',
    'volume',
    'sma_20',
    'rsi_14',
    'macd_12_26_9'
]
TARGET_COLUMN = 'close' # predicted