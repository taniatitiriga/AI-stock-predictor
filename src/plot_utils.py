import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import config

def plot_future_prediction(historical_data_df, look_back_window, predicted_date, predicted_value, ticker_symbol, target_column_name='close'):
    plt.figure(figsize=(15, 7))

    plot_context_days = look_back_window + 90
    if len(historical_data_df) > plot_context_days:
        plot_data = historical_data_df.iloc[-plot_context_days:]
    else:
        plot_data = historical_data_df

    plt.plot(plot_data.index, plot_data[target_column_name], label=f'Historical Actual {target_column_name}', color='blue')

    predicted_datetime = pd.to_datetime(predicted_date)
    plt.scatter([predicted_datetime], [predicted_value], color='red', marker='o', s=100, label=f'Predicted {target_column_name} for {predicted_date}', zorder=5)

    plt.text(predicted_datetime, predicted_value, f'{predicted_value:.2f}', color='red', ha='left', va='bottom')


    plt.title(f'{ticker_symbol} - {target_column_name.capitalize()} prediction [BETA]')
    plt.xlabel('Date')
    plt.ylabel(f'{target_column_name.capitalize()} Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def plot_predictions(original_data_df, look_back, y_train_actual, train_predict, y_test_actual, test_predict, ticker_symbol):
    plt.figure(figsize=(15, 7))

    date_index = original_data_df.index

    # plot training
    train_plot_idx_end = look_back + len(y_train_actual)
    plt.plot(date_index[look_back:train_plot_idx_end],
             y_train_actual.flatten(),
             label='Actual Train Price', color='blue')
    plt.plot(date_index[look_back:train_plot_idx_end],
             train_predict.flatten(),
             label='Predicted Train Price', color='orange', linestyle='--')

    # plot testing
    test_plot_idx_start = train_plot_idx_end
    test_plot_idx_end = test_plot_idx_start + len(y_test_actual)
    plt.plot(date_index[test_plot_idx_start:test_plot_idx_end],
             y_test_actual.flatten(),
             label='Actual Test Price', color='green')
    plt.plot(date_index[test_plot_idx_start:test_plot_idx_end],
             test_predict.flatten(),
             label='Predicted Test Price', color='red', linestyle='--')
    
    plt.title(f'{ticker_symbol} Stock Price Prediction (LSTM)')
    plt.xlabel('Date')
    plt.ylabel(f'{config.TARGET_COLUMN} Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss(history):
    """Draws training and validation loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_weekly_predictions(
    hist_df,
    wk_preds_list,
    tkr_sym,
    tgt_col_name='close'
):
    if not wk_preds_list:
        print(f"No weekly predictions to plot for {tkr_sym}.")
        return

    plt.figure(figsize=(15, 7))

    first_pred_date_obj = pd.to_datetime(wk_preds_list[0]['date'])
    
    # --- MODIFIED: Select historical data for ~1 month context ---
    # Define how many trading days roughly constitute a month (e.g., 20-22)
    # Or use a fixed number of calendar days, e.g., 30.
    # Let's aim for approximately 30 calendar days of history before the first prediction.
    hist_context_days_approx = 30 
    # Calculate the start date for the historical plot segment
    hist_plot_start_date_target = first_pred_date_obj - pd.Timedelta(days=hist_context_days_approx)
    
    # Select historical data points that fall within this targeted historical window
    # and are before the first prediction.
    plot_hist_data_context = hist_df[
        (hist_df.index >= hist_plot_start_date_target) & 
        (hist_df.index < first_pred_date_obj)
    ].copy()
    
    # Ensure there's at least *some* historical data if the above filter is too strict
    # or if the prediction starts very close to the beginning of hist_df.
    if plot_hist_data_context.empty and not hist_df[hist_df.index < first_pred_date_obj].empty:
        # Fallback to last N actual historical points if the date range filter yields nothing
        # but there is data before the prediction.
        num_fallback_points = 20 # Show at least 20 points if possible
        plot_hist_data_context = hist_df[hist_df.index < first_pred_date_obj].iloc[-num_fallback_points:].copy()


    # Plot historical actual prices
    if not plot_hist_data_context.empty and tgt_col_name in plot_hist_data_context.columns:
        plt.plot(plot_hist_data_context.index, plot_hist_data_context[tgt_col_name], label=f'Historical {tgt_col_name}', color='blue', zorder=1)
        last_hist_date = plot_hist_data_context.index[-1]
        last_hist_value = plot_hist_data_context[tgt_col_name].iloc[-1]
    else:
        print(f"Warning: Not enough/no historical data or target column '{tgt_col_name}' missing for weekly plot context for {tkr_sym}.")
        last_hist_date = None
        last_hist_value = None

    # ... (The rest of the function for preparing and plotting predictions remains THE SAME as before) ...
    pred_dates_dt = [pd.to_datetime(p['date']) for p in wk_preds_list]
    pred_values_num = [p['prediction'] for p in wk_preds_list]

    plot_pred_line_dates = []
    plot_pred_line_values = []

    if last_hist_date is not None and last_hist_value is not None:
        plot_pred_line_dates.append(last_hist_date)
        plot_pred_line_values.append(last_hist_value)
    
    plot_pred_line_dates.extend(pred_dates_dt)
    plot_pred_line_values.extend(pred_values_num)
    
    if plot_pred_line_dates:
        plt.plot(plot_pred_line_dates, plot_pred_line_values, color='red', linestyle='--', label=f'Predicted {tgt_col_name} (Next Week)', zorder=2)

    if pred_dates_dt:
        plt.scatter(pred_dates_dt, pred_values_num, color='red', marker='o', s=30, zorder=3)

    for date_pt, val_pt in zip(pred_dates_dt, pred_values_num):
        plt.text(date_pt, val_pt, f'{val_pt:.2f}', color='red', ha='left', va='bottom', fontsize=8)

    plt.title(f'{tkr_sym} - Next Week {tgt_col_name.capitalize()} Prediction ({len(wk_preds_list)} days)')
    plt.xlabel('Date')
    plt.ylabel(f'{tgt_col_name.capitalize()} Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_model(y_true, y_pred, set_name="Test"):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mask = y_true_flat != 0
    if np.sum(mask) == 0: # avoid 0 in MAPE (bad results)
        mape = np.inf
    else:
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100

    rmse = np.sqrt(np.mean((y_pred_flat - y_true_flat)**2))

    print(f"{set_name} RMSE: {rmse:.4f}")
    print(f"{set_name} MAPE: {mape:.2f}%")
    return mape 