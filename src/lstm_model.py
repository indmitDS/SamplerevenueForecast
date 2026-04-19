import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self, data, look_back=1):
        self.data = data
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler()  # Initialize the scaler
        self.X, self.Y = self.prepare_data(look_back)

    def prepare_data(self, look_back=1):
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))

        X, Y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:(i + look_back)])
            Y.append(scaled_data[i + look_back])
        return np.array(X).reshape(-1, look_back, 1), np.array(Y)

    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self, epochs=100, batch_size=1):
        self.build_model(input_shape=(self.look_back, 1))
        self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, verbose=2)

    def get_fitted_values(self):
        # Get fitted values (in-sample predictions)
        fitted_scaled = self.model.predict(self.X).flatten()
        # Inverse transform to revert back to original scale
        return self.scaler.inverse_transform(fitted_scaled.reshape(-1, 1)).flatten()

    def forecast(self, steps=80):
        # Use the last 'look_back' points to start forecasting
        last_sequence = self.data[-self.look_back:].values
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))

        predictions = []
        for _ in range(steps):
            prediction = self.model.predict(last_sequence_scaled.reshape(1, -1, 1))
            predictions.append(prediction[0][0])
            last_sequence_scaled = np.append(last_sequence_scaled[1:], prediction[0][0])

        # Inverse transform to revert back to original scale
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()










