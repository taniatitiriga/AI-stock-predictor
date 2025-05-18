import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from src import config
from src.data_utils import fetch_stock_data, preprocess_for_lstm
from src.model_utils import build_lstm_model, train_lstm_model, make_predictions, inverse_transform_predictions
from src.plot_utils import plot_predictions, plot_loss, evaluate_model, plot_future_prediction, plot_weekly_predictions
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
        
        raw_data_ta.ffill(inplace=True)
        raw_data_ta.bfill(inplace=True)
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

    plot_predictions(raw_data, config.LOOK_BACK_WINDOW, y_train_actual, train_predict, y_test_actual, test_predict, f"{ticker_symbol}")
    # plot_loss(history, f"{ticker_symbol}_eval_loss")

    print("Done!")
    
    return test_mape, eval_model, scaler, scaled_cols_names, target_idx, raw_data_ta

def predict_next_day(ticker_symbol, historical_ta):
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

    print(f"Predicted {config.TARGET_COLUMN} for {ticker_symbol} on {prediction_date_str}: {predicted_value:.2f}")
    
    plot_future_prediction(
        historical_ta,
        config.LOOK_BACK_WINDOW,
        prediction_date_str,
        predicted_value,
        ticker_symbol,
        config.TARGET_COLUMN
    )
    return prediction_date_str, predicted_value

def predict_next_week(ticker_symbol, historical_ta_input, nr_days=5):
    print(f"\n Making predictions for the next {nr_days} trading days for {ticker_symbol}...")

    if historical_ta_input is None or historical_ta_input.empty:
        print(f"No historical data provided for {ticker_symbol} to predict next week.")
        return []

    historical_ta_for_model = historical_ta_input.copy() # copy for model training
    
    scaler, X_all_initial, _, y_all_initial, _, scaled_col_names, target_idx = preprocess_for_lstm(
        historical_ta_for_model.copy(),
        config.FEATURES_TO_USE,
        config.TARGET_COLUMN,
        config.LOOK_BACK_WINDOW,
        train_split_ratio=1.0 
    )

    if X_all_initial.size == 0:
        print(f"Insufficient data to train model for weekly prediction for {ticker_symbol}.")
        return []

    input_shape = (X_all_initial.shape[1], X_all_initial.shape[2])
    model = build_lstm_model(input_shape)

    print(f"Training model on all historical data for {ticker_symbol} (for weekly prediction)...")
    model.fit(
        X_all_initial, y_all_initial,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        verbose=1 # customizable
    )

    # iterate next week
    predictions_for_week = []
    curr_last_data_ta = historical_ta_input.copy() # Data to append predictions to for TA recalculation
    curr_last_seq_scaled = X_all_initial[-1:, :, :] # Last sequence from initial scaled data

    calendar = mcal.get_calendar('NYSE')

    for i in range(nr_days):
        # determine next trading day
        last_date = curr_last_data_ta.index[-1]
        
        sched = calendar.schedule(
            start_date=last_date.strftime('%Y-%m-%d'),
            end_date=(last_date + timedelta(days=15)).strftime('%Y-%m-%d') # look ahead further (bug fix?)
        )
        prediction_dates = sched.index.normalize()
        target_date = None
        for potential_date in prediction_dates:
            if potential_date > last_date:
                target_date = potential_date
                break
        
        if target_date is None:
            print(f"Warning: Could not find next trading day after {last_date.strftime('%Y-%m-%d')}. Stopping weekly prediction.")
            break 
        
        target_date_str = target_date.strftime('%Y-%m-%d')

        # predict using current last sequence scaled
        predicted_scaled = model.predict(curr_last_seq_scaled)
        predicted_actual = inverse_transform_predictions(predicted_scaled, scaler, scaled_col_names, target_idx)[0,0]

        print(f"Day {i+1}/{nr_days}: Predicted {config.TARGET_COLUMN} for {target_date_str}: {predicted_actual:.2f}")
        predictions_for_week.append({"date": target_date_str, "prediction": predicted_actual})

        # prepare next
        if i < nr_days - 1:
            # create synthetic row for predicted day
            row_feat = {col: np.nan for col in config.FEATURES_TO_USE}
            # fill OHLC with predicted close, vol. - last known
            row_feat['open'] = predicted_actual
            row_feat['high'] = predicted_actual 
            row_feat['low'] = predicted_actual
            row_feat[config.TARGET_COLUMN] = predicted_actual
            if 'volume' in row_feat and not curr_last_data_ta['volume'].empty:
                 row_feat['volume'] = curr_last_data_ta['volume'].iloc[-1]

            new_row_df = pd.DataFrame([row_feat], index=[target_date])
            
            # append new row to data
            curr_last_data_ta = pd.concat([curr_last_data_ta, new_row_df])
            
            # recalculate TA
            if 'volume' in curr_last_data_ta.columns:
                ta_df_temp = curr_last_data_ta.copy() # copy for TA
                

                ta_cols_to_drop = [col for col in ta_df_temp.columns if 'sma_' in col or 'rsi_' in col or 'macd_' in col or 'macds_' in col or 'macdh_' in col]
                ta_df_temp.drop(columns=ta_cols_to_drop, inplace=True, errors='ignore')

                ta_df_temp.ta.sma(length=20, append=True)
                ta_df_temp.ta.rsi(length=14, append=True)
                ta_df_temp.ta.macd(append=True)
                ta_df_temp.columns = ta_df_temp.columns.str.lower().str.strip() # clean names
                ta_df_temp.ffill(inplace=True) # fill NaNs from new TA calcs
                ta_df_temp.bfill(inplace=True)
                curr_last_data_ta = ta_df_temp # update with new TAs


            updated_last_window = curr_last_data_ta[scaled_col_names].iloc[-config.LOOK_BACK_WINDOW:]
            
            if len(updated_last_window) < config.LOOK_BACK_WINDOW:
                print("Warning: Not enough data for look_back after appending prediction. Stopping.")
                break
            
            # use initial scaler
            current_last_sequence_scaled_array = scaler.transform(updated_last_window)
            curr_last_seq_scaled = np.expand_dims(current_last_sequence_scaled_array, axis=0)
    
    # draw
    if predictions_for_week:
        plot_weekly_predictions(
            historical_ta_input,
            predictions_for_week,
            ticker_symbol,
            config.TARGET_COLUMN
        )
    
    return predictions_for_week

if __name__ == "__main__":
    tickers = []
    
    benchmark_mapes = {}
    next_day_pred = {} 
    next_week_pred = {} 
    
    # flags
    RUN_BENCHMARK = False
    PREDICT_DAY = False
    PREDICT_WEEK = False
    NR_DAYS = 5
    
    print("\n\nAI Stock Predictor\n")
    print("This project uses LSTM (long short-term memory) neural network to perform time series analysis and forecasting. Historical data from Yahoo Finance.")
    print("Disclaimer: the predictions made are based purely on historical data and mathematical analysis, therefore do not account for political or other external factors. This tool was build for academic purposes only.\n\n")
    
    quitted = False
    while not quitted:
        print(f"Current pick: {tickers}.")
        ticker = input("Please input a valid ticker (e.g. SPLV - low in noise) or type 'continue': ").strip().upper()
        if ticker == "CONTINUE":
            ticket = None
            quitted = True
        elif ticker.isalnum():
            tickers.append(ticker)
        else:
            print("Invalid ticker, please try again.")
    
    print(f"Final list: {tickers}.\n")
    
    print("Pick the feature(s) you wish to use:")
    print("1. Benchmark - test the model on historical data and compare it to actual data. Errors measured with MAPE.")
    RUN_BENCHMARK = input("Select feature? (y/n):").strip().lower() == "y"
    print("2. Predict tomorrow - train the model on all available historical data and guess tomorrow's closing price.")
    PREDICT_DAY = input("Select feature? (y/n):").strip().lower() == "y"
    print("3. Predict next week - train the model on all available historical data and guess next week's closing prices (5 NYSE market days).")
    PREDICT_WEEK = input("Select feature? (y/n):").strip().lower() == "y"
    
    print("\nThis may take a while - the prototype is still yet to be optimized :) \n\n")
    
    for ticker in tickers:
        print(f"========== PROCESSING {ticker} ==========")
        historical_data_with_ta = None

        if RUN_BENCHMARK:
            mape, _, _, _, _, data_df_from_benchmark = benchmark(ticker)
            if mape is not None:
                benchmark_mapes[ticker] = mape
            if data_df_from_benchmark is not None:
                historical_data_with_ta = data_df_from_benchmark
        else:
            # fetch data directly for prediction phases
            print(f"Fetching data for {ticker}...")
            
            raw_data_for_predict = fetch_stock_data(ticker, config.START_DATE, config.END_DATE)
            if raw_data_for_predict is not None and not raw_data_for_predict.empty:
                temp_data_ta = raw_data_for_predict.copy()
                
                if 'volume' in temp_data_ta.columns:
                    temp_data_ta.ta.sma(length=20, append=True)
                    temp_data_ta.ta.rsi(length=14, append=True)
                    temp_data_ta.ta.macd(append=True)
                    
                    temp_data_ta.columns = temp_data_ta.columns.str.lower().str.strip()
                    temp_data_ta.ffill(inplace=True)
                    temp_data_ta.bfill(inplace=True)
                    
                    historical_data_with_ta = temp_data_ta
                    
                else:
                    print(f"'volume' column not found for {ticker}, TA indicators skipped for direct prediction.")
                    historical_data_with_ta = raw_data_for_predict 
            else:
                print(f"Could not fetch data for {ticker} for prediction phases.")


        if PREDICT_DAY:
            if historical_data_with_ta is not None:
                pred_date, pred_value = predict_next_day(ticker, historical_data_with_ta) # MODIFIED: Using renamed function
                if pred_date is not None:
                    next_day_pred[ticker] = {"date": pred_date, "prediction": pred_value}
            else:
                print(f"Next day prediction for {ticker} - missing historical data.")
        
        if PREDICT_WEEK:
            if historical_data_with_ta is not None:
                weekly_preds = predict_next_week(ticker, historical_data_with_ta, NR_DAYS)
                if weekly_preds: # If list is not empty
                    next_week_pred[ticker] = weekly_preds
            else:
                print(f"Next week prediction for {ticker} - missing historical data.")


    if RUN_BENCHMARK:
        print("\n Benchmark results:")
        for ticker, mape_val in benchmark_mapes.items():
            print(f"{ticker} test MAPE = {mape_val:.2f}%")

    if PREDICT_DAY: 
        print(f"\nSingle Next Day Prediction summary (1 trading day after {config.END_DATE})")
        
        for ticker, pred_info in next_day_pred.items():
            print(f"  {ticker} \n  Date: {pred_info['date']}, Predicted {config.TARGET_COLUMN} = {pred_info['prediction']:.2f}")
    
    if PREDICT_WEEK:
        print(f"\nUpcoming Week Prediction summary ({NR_DAYS} trading days after {config.END_DATE})")
        for ticker, weekly_preds_list in next_week_pred.items():
            print(f"{ticker}:")
            for pred_info in weekly_preds_list:
                 print(f"Date: {pred_info['date']}, Predicted {config.TARGET_COLUMN} = {pred_info['prediction']:.2f}")