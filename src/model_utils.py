import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from . import config

def build_lstm_model(input_shape):
    # input_shape is (look_back, num_features)
    model = Sequential()
    
    # input layer
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # Dropout for regularization

    # hidden layers
    model.add(LSTM(units=80, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # dense layer
    model.add(Dense(units=25, activation='relu'))

    # output layer
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    print("Training the model...")
    
    # prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def make_predictions(model, X_data):
    print("Making predictions...")
    return model.predict(X_data)

def inverse_transform_predictions(predictions_scaled, scaler, all_scaled_columns, target_col_idx):
    num_features_scaled = len(all_scaled_columns)
    
    # model columns for nr of features
    dummy_array = np.zeros((len(predictions_scaled), num_features_scaled))
    dummy_array[:, target_col_idx] = predictions_scaled.flatten()
    
    # inverse transform
    inversed = scaler.inverse_transform(dummy_array)
    return inversed[:, target_col_idx].reshape(-1, 1)