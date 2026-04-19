import logging
import pandas as pd
import os
import pickle

class Forecasting:
    def __init__(self, models, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = models
        self.config = config

    def forecast_arima(self):
        self.logger.debug("Generating forecast using ARIMA model.")
        try:
            model = self.models['ARIMA']
            forecast = model.forecast(steps=self.config['forecast']['forecast_periods'])
            forecast_df = pd.DataFrame(forecast, columns=['ARIMA_Forecast'])
            forecast_path = os.path.join(self.config['paths']['results'], 'arima_forecast.csv')
            forecast_df.to_csv(forecast_path, index=False)
            self.logger.info("ARIMA forecast generated and saved.")
            return forecast_df
        except Exception as e:
            self.logger.error(f"Error generating ARIMA forecast: {e}")
            raise

    def forecast_prophet(self):
        self.logger.debug("Generating forecast using Prophet model.")
        try:
            future = self.models['Prophet'].make_future_dataframe(periods=self.config['forecast']['forecast_periods'])
            forecast = self.models['Prophet'].predict(future)
            forecast_df = forecast[['ds', 'yhat']]
            forecast_path = os.path.join(self.config['paths']['results'], 'prophet_forecast.csv')
            forecast_df.to_csv(forecast_path, index=False)
            self.logger.info("Prophet forecast generated and saved.")
            return forecast_df
        except Exception as e:
            self.logger.error(f"Error generating Prophet forecast: {e}")
            raise

    def forecast_xgboost(self, X_future):
        self.logger.debug("Generating forecast using XGBoost model.")
        try:
            forecast = self.models['XGBoost'].predict(X_future)
            forecast_df = pd.DataFrame(forecast, columns=['XGBoost_Forecast'])
            forecast_path = os.path.join(self.config['paths']['results'], 'xgboost_forecast.csv')
            forecast_df.to_csv(forecast_path, index=False)
            self.logger.info("XGBoost forecast generated and saved.")
            return forecast_df
        except Exception as e:
            self.logger.error(f"Error generating XGBoost forecast: {e}")
            raise