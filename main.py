import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from src import config
from src.data_utils import fetch_stock_data, preprocess_for_lstm
from src.model_utils import build_lstm_model, train_lstm_model, make_predictions, inverse_transform_predictions
from src.plot_utils import plot_predictions, plot_loss, evaluate_model, plot_future_prediction
import pandas_ta as ta

def benchmark(ticker_symbol):
    print(f"Testing accuracy for {ticker_symbol}")
    
     # fetch data from Yahoo Finance API
    raw_data = fetch_stock_data(ticker_symbol, config.START_DATE, config.END_DATE)

    if raw_data is None or raw_data.empty:
        print(f"No data for {ticker_symbol}")
        return None, None, None, None, None, None

    # adjust technical indicators
    raw_data_ta = raw_data.copy()
    if 'volume' in raw_data_ta.columns:
        raw_data_ta.ta.sma(length=20, append=True)
        raw_data_ta.ta.rsi(length=14, append=True)
        raw_data_ta.ta.macd(append=True)

        raw_data_ta.columns = raw_data_ta.columns.str.lower().str.strip()
        
        raw_data_ta.fillna(method='ffill', inplace=True)
        raw_data_ta.fillna(method='bfill', inplace=True)
    else:
        print(f"'volume' column not found for {ticker_symbol}")
    
    # pre-process data
    scaler, X_train, X_test, y_train, y_test, scaled_cols_names, target_idx = preprocess_for_lstm(
        raw_data_ta.copy(),
        config.FEATURES_TO_USE,
        config.TARGET_COLUMN,
        config.LOOK_BACK_WINDOW,
        config.TRAIN_SPLIT_RATIO
    )
    
    if X_train.size == 0 or X_test.size == 0:
        print(f"Insufficient data for {ticker_symbol}")
        return None, None, None, None, None, raw_data_ta
    
    # build evaluation model
    input_shape = (X_train.shape[1], X_train.shape[2])
    eval_model = build_lstm_model(input_shape)

    # train
    print("Training...")
    history = train_lstm_model(
        eval_model, X_train, y_train, X_test, y_test,
        config.EPOCHS, config.BATCH_SIZE
    )

    # test predictions
    train_predict_scaled = make_predictions(eval_model, X_train)
    test_predict_scaled = make_predictions(eval_model, X_test)

    # inverse transform
    train_predict = inverse_transform_predictions(train_predict_scaled, scaler, scaled_cols_names, target_idx)
    test_predict = inverse_transform_predictions(test_predict_scaled, scaler, scaled_cols_names, target_idx)
    y_train_actual = inverse_transform_predictions(y_train.reshape(-1,1), scaler, scaled_cols_names, target_idx)
    y_test_actual = inverse_transform_predictions(y_test.reshape(-1,1), scaler, scaled_cols_names, target_idx)

    # benchmark
    print(f"\n MAPE for {ticker_symbol} -up to {config.END_DATE}")
    _ = evaluate_model(y_train_actual, train_predict, "Train")
    test_mape = evaluate_model(y_test_actual, test_predict, "Test")

    plot_predictions(raw_data, config.LOOK_BACK_WINDOW, y_train_actual, train_predict, y_test_actual, test_predict, f"{ticker_symbol}_eval")
    # plot_loss(history, f"{ticker_symbol}_eval_loss")

    print("Done!")
    
    return test_mape, eval_model, scaler, scaled_cols_names, target_idx, raw_data_ta

def predict(ticker_symbol, historical_ta):
    print(f"\n Making predictions for {ticker_symbol}...")

    if historical_ta is None or historical_ta.empty:
        print(f"No data for {ticker_symbol}")
        return None, None

    scaler, X_all, _, y_all, _, scaled_col_names, target_idx = preprocess_for_lstm(
        historical_ta.copy(),
        config.FEATURES_TO_USE,
        config.TARGET_COLUMN,
        config.LOOK_BACK_WINDOW,
        train_split_ratio=1.0 # use all data for training
    )

    if X_all.size == 0:
        print(f"Insufficient data for {ticker_symbol}")
        return None, None

    # build final model
    input_shape = (X_all.shape[1], X_all.shape[2])
    model = build_lstm_model(input_shape)

    # training
    print(f"Re-training for {ticker_symbol}")
    model.fit(
        X_all, y_all,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1
    )

    last_sequence_scaled = X_all[-1:, :, :] # (1, look_back_window, num_features)

    # predict
    predicted_next_day_scaled = model.predict(last_sequence_scaled)

    # inverse transform the prediction (unscale)
    predicted_next_day_actual = inverse_transform_predictions(
        predicted_next_day_scaled, scaler, scaled_col_names, target_idx
    )
    
    predicted_value = predicted_next_day_actual[0,0]
    
    # find next market day
    last_known_date = historical_ta.index[-1]
    exchange_calendar = mcal.get_calendar('NYSE')
    
    sched = exchange_calendar.schedule(
        start_date=last_known_date.strftime('%Y-%m-%d'), 
        end_date=(last_known_date + timedelta(days=7)).strftime('%Y-%m-%d')
    )

    pot_date = sched.index.normalize()
    
    prediction_date = None
    for potential_date in pot_date:
        if potential_date > last_known_date:
            prediction_date = potential_date
            break
            
    if prediction_date is None:
        print("Couldn't find next trading day, using default")
        prediction_date = last_known_date + timedelta(days=1)
    
    prediction_date_str = prediction_date.strftime('%Y-%m-%d')

    print(f"Predicted {config.TARGET_COLUMN} for {ticker_symbol} on {prediction_date.strftime('%Y-%m-%d')}: {predicted_value:.2f}")
    
    plot_future_prediction(
        historical_ta,
        config.LOOK_BACK_WINDOW,
        prediction_date_str,
        predicted_value,
        ticker_symbol,
        config.TARGET_COLUMN
    )
    return prediction_date_str, predicted_value


if __name__ == "__main__":
    tickers = ['SPLV'] # lower noise example

    benchmark_mape = {}
    predictions = {}

    # TBD customizable
    to_predict = True

    for ticker in tickers:
        mape, model, scaler, s_cols, t_idx, data_df = benchmark(ticker)

        if mape is not None:
            benchmark_mape[ticker] = mape

        if to_predict and data_df is not None:
            pred_date, pred_value = predict(ticker, data_df)
            if pred_date is not None:
                predictions[ticker] = {"date": pred_date, "prediction": pred_value}
        elif to_predict:
            print(f"Insufficient data for {ticker}")


    print("\n Benchmark results:")
    for ticker, mape_val in benchmark_mape.items():
        print(f"{ticker} test MAPE = {mape_val:.2f}%")

    if to_predict:
        print(f"\nPrediction summary (1 day after {config.END_DATE})")
        for ticker, pred_info in predictions.items():
            print(f"{ticker} \nDate: {pred_info['date']}, Predicted {config.TARGET_COLUMN} = {pred_info['prediction']:.2f}")