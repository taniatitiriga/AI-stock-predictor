from src import config
from src.data_utils import fetch_stock_data, preprocess_for_lstm
from src.model_utils import build_lstm_model, train_lstm_model, make_predictions, inverse_transform_predictions
from src.plot_utils import plot_predictions, plot_loss, evaluate_model
import pandas_ta as ta

def run_pipeline(ticker_symbol):
    # fetch data from Yahoo Finance API
    raw_data_df = fetch_stock_data(ticker_symbol, config.START_DATE, config.END_DATE)
    
    if raw_data_df is None or raw_data_df.empty:
        print(f"No data for {ticker_symbol}, skipping.")
        return
    
    # adjust number of features to analyse
    if 'volume' in raw_data_df.columns:
        raw_data_df.ta.sma(length=20, append=True)
        raw_data_df.ta.rsi(length=14, append=True)
        raw_data_df.ta.macd(append=True)

        raw_data_df.columns = raw_data_df.columns.str.lower()
        raw_data_df.columns = raw_data_df.columns.str.strip()

        raw_data_df.fillna(method='bfill', inplace=True)
        raw_data_df.fillna(method='ffill', inplace=True)
    else:
        print("Warning: 'volume' column (lowercase) not found. Skipping TA indicators or they might fail.")

    # pre-process data
    scaler, X_train, X_test, y_train, y_test, scaled_cols_names, target_idx = preprocess_for_lstm(
        raw_data_df.copy(),
        config.FEATURES_TO_USE,
        config.TARGET_COLUMN,
        config.LOOK_BACK_WINDOW,
        config.TRAIN_SPLIT_RATIO
    )

    # build - input_shape for LSTM is (timesteps, num_features)
    model_input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(model_input_shape)
    
    # train
    history = train_lstm_model(
        model, X_train, y_train, X_test, y_test,
        config.EPOCHS, config.BATCH_SIZE
    )
    
    #predict
    train_predict_scaled = make_predictions(model, X_train)
    test_predict_scaled = make_predictions(model, X_test)

    # inverse transform based on scale
    train_predict_actual = inverse_transform_predictions(train_predict_scaled, scaler, scaled_cols_names, target_idx)
    test_predict_actual = inverse_transform_predictions(test_predict_scaled, scaler, scaled_cols_names, target_idx)
    
    y_train_actual_flat = y_train.reshape(-1,1) # y_train 2D for inverse_transform_predictions
    y_train_actual = inverse_transform_predictions(y_train_actual_flat, scaler, scaled_cols_names, target_idx)

    y_test_actual_flat = y_test.reshape(-1,1) # y_test 2D for inverse_transform_predictions
    y_test_actual = inverse_transform_predictions(y_test_actual_flat, scaler, scaled_cols_names, target_idx)

    # eval
    evaluate_model(y_train_actual, train_predict_actual, "Train")
    evaluate_model(y_test_actual, test_predict_actual, "Test")

    # draw
    plot_predictions(
        raw_data_df, config.LOOK_BACK_WINDOW,
        y_train_actual, train_predict_actual,
        y_test_actual, test_predict_actual,
        ticker_symbol
    )
    plot_loss(history)

    print(f"{ticker_symbol} finished.")

if __name__ == "__main__":
    tickers_to_test = ['MSFT', 'GOOGL', 'AMZN', 'TSLA']

    all_test_rmses = {}
    for ticker in tickers_to_test:
        try:
            test_rmse = run_pipeline(ticker) 
            if test_rmse is not None:
                all_test_rmses[ticker] = test_rmse
        except Exception as e:
            print(f"ERROR processing {ticker}: {e}")
    
    # print("\n--- SUMMARY OF TEST RMSES ---")
    # for ticker, rmse in all_test_rmses.items():
    #     print(f"{ticker}: Test RMSE = {rmse:.4f}")