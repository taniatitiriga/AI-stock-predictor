import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from . import config

def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    data_df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by='ticker'
    )
    if data_df.empty:
        print(f"No data found for {ticker}. Exiting.")
        exit()
        
    if isinstance(data_df.columns, pd.MultiIndex):
        
        if len(data_df.columns.levels) > 1 and data_df.columns.levels[0].name == 'Ticker' and len(data_df.columns.levels[0]) == 1:
            data_df = data_df[ticker]
        elif len(data_df.columns.levels) > 1 :
             data_df.columns = data_df.columns.get_level_values(-1)
        else:
            if len(data_df.columns.levels) > 0:
                 data_df.columns = data_df.columns.get_level_values(-1)
    
    data_df.columns = data_df.columns.astype(str)

    data_df.columns = data_df.columns.str.lower()
    data_df.columns = data_df.columns.str.strip()

    if 'adj close' in data_df.columns:
        data_df.rename(columns={'adj close': 'adj_close'}, inplace=True)

    return data_df

def preprocess_for_lstm(data_df, features_to_use, target_column, look_back, train_split_ratio):
    # target_column in features_to_use if it's not already
    all_columns_for_scaling = list(set(features_to_use + [target_column]))
    
    # set dataframe cols
    model_data = data_df[all_columns_for_scaling].copy()
    
    if model_data.isnull().values.any():
        print("Warning: NaN values found in data. Filling with previous values (ffill)...")
        model_data.fillna(method='ffill', inplace=True)
        model_data.fillna(method='bfill', inplace=True)

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(model_data)

    try:
        target_col_index = model_data.columns.get_loc(target_column)
    except KeyError:
        print(f"Error: Target column '{target_column}' not found in the selected features for scaling.")
        exit()

    X, y = [], []
    for i in range(look_back, len(scaled_data)): 
        # X - 'look_back' nr of previous days data
        X.append(scaled_data[i-look_back:i, :]) 
        
        # y - target variable for predicted day
        y.append(scaled_data[i, target_col_index])

    X, y = np.array(X), np.array(y)

    # split data
    training_size = int(len(X) * train_split_ratio)
    
    X_train, X_test = X[0:training_size], X[training_size:len(X)]
    y_train, y_test = y[0:training_size], y[training_size:len(y)]
 
    return scaler, X_train, X_test, y_train, y_test, model_data.columns.tolist(), target_col_index