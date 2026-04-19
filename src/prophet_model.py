from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class ProphetModel:
    def __init__(self, data):
        if isinstance(data, pd.Series):
            self.data = data.reset_index()
            self.data.columns = ['ds', 'y']
        else:
            self.data = data.rename(columns={'Date': 'ds', 'TOTAL_REVENUE_USD': 'y'})

        # Initialize the scaler
        self.scaler = MinMaxScaler()

        # Scale the target variable
        self.data['y'] = self.scaler.fit_transform(self.data[['y']])

        self.model = Prophet(interval_width=0.95, 
                             yearly_seasonality=True, weekly_seasonality=True,
                             changepoint_prior_scale=0.5, 
                             seasonality_prior_scale=0.1)

    def fit(self):
        self.model.fit(self.data)

    def predict(self, periods=80):
       # Create future dates starting from the last date in historical data
       future = self.model.make_future_dataframe(periods=periods, freq='W')

       # Predict the future values
       forecast = self.model.predict(future)

        # Filter the forecast to only include the number of desired periods
       last_date = self.data['ds'].max()
       future_dates = pd.date_range(last_date, periods=periods, freq='W')
       forecast_filtered = forecast[forecast['ds'].isin(future_dates)]

       # Revert the scaling on the forecasted values
       forecast_filtered[['yhat', 'yhat_lower', 'yhat_upper']] = self.scaler.inverse_transform(
        forecast_filtered[['yhat', 'yhat_lower', 'yhat_upper']])

        # Set the index to the 'ds' column for better alignment with original data
       forecast_filtered.set_index('ds', inplace=True)

           # Returning forecasted values and confidence intervals
       return forecast_filtered[['yhat', 'yhat_lower', 'yhat_upper']]


    def get_fitted_values(self):
        # Fitted values (in-sample predictions)
        forecast = self.model.predict(self.data)

        # Revert scaling on the fitted values
        fitted_values = self.scaler.inverse_transform(forecast[['yhat']])

        return pd.Series(fitted_values.flatten(), index=self.data['ds'])