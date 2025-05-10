import matplotlib.pyplot as plt
import numpy as np
from . import config

def plot_predictions(original_data_df, look_back, 
                     y_train_actual, train_predict, 
                     y_test_actual, test_predict, 
                     ticker_symbol):
    """Draw actual vs. predicted prices"""
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

def evaluate_model(y_true, y_pred, set_name="Test"):
    """Compute RMSE"""
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    print(f"{set_name} RMSE: {rmse:.4f}")
    return rmse