import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml
from scipy.interpolate import make_interp_spline
from src.prophet_model import ProphetModel
from src.lstm_model import LSTMModel
from src.sarima_model import SARIMAModel

class ModelEvaluation:

    def __init__(self, df, config_path):
        self.df = df
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.results = {}

    def evaluate_model(self, model, model_name):

      if isinstance(model, ProphetModel):
        # Prophet-specific prediction
        print("Prophet identified")
        forecast_data = model.predict(periods=80)
        forecast = forecast_data['yhat']
        conf_int = forecast_data[['yhat_lower', 'yhat_upper']]

        fitted = model.get_fitted_values()

      elif isinstance(model, LSTMModel):
        # LSTM-specific prediction 
        print("LSTMModel identified")
        fitted = model.get_fitted_values()
        forecast = model.forecast(steps=80)
        conf_int = pd.DataFrame({
            'yhat_lower': forecast - 1.96 * np.std(fitted),
            'yhat_upper': forecast + 1.96 * np.std(fitted)
        })

        fitted = pd.Series(fitted, index=self.df.index[:len(fitted)])

      elif isinstance(model, SARIMAModel):
        # SARIMA-specific prediction
        print("SARIMA Model identified")
        forecast, conf_int = model.predict(steps=80)

        # Check if the model has a get_fitted_values method
        if hasattr(model, 'get_fitted_values'):
            fitted = model.get_fitted_values()
        else:
            fitted = model.model.fittedvalues

      else:
        # ARIMA
        print("ARIMA/Other Model identified")
        forecast, conf_int = model.predict(steps=80)

        if hasattr(model, 'get_fitted_values'):
            fitted = model.get_fitted_values()
        else:
            fitted = model.model.fittedvalues

       # Ensure both indices are datetime
      if not isinstance(fitted.index, pd.DatetimeIndex):
        fitted.index = pd.to_datetime(fitted.index)
      if not isinstance(self.df.index, pd.DatetimeIndex):
        self.df.index = pd.to_datetime(self.df.index)

      # Align fitted values with the actual data's index
      fitted = fitted.reindex(self.df.index, method='nearest')


      residuals = self.df - fitted

        # Save results
      self.results[model_name] = {
        'forecast': forecast,
        'conf_int': conf_int,
        'fitted': fitted,
        'residuals': residuals
          }

        # Ensure lengths match before calculating metrics
      if len(self.df) != len(fitted):
        print(f"Error: Mismatch in lengths. Actual: {len(self.df)}, Fitted: {len(fitted)}")
        return  # Stop evaluation if lengths don't match

      # Calculate evaluation metrics
      mse = mean_squared_error(self.df, fitted)
      mae = mean_absolute_error(self.df, fitted)
      self.results[model_name]['mse'] = mse
      self.results[model_name]['mae'] = mae

       # Print summary
      print(f"Model: {model_name}")
      print(f"Mean Squared Error: {mse}")
      print(f"Mean Absolute Error: {mae}")

       # Save summary results
      with open(f"{self.config['paths']['results']}/{model_name}_summary.txt", "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Mean Squared Error: {mse}\n")
        file.write(f"Mean Absolute Error: {mae}\n")


    def smooth_plot(self, x, y, label, color):
         # Convert week-based datetime to numeric index (e.g., number of weeks since start)
         x_numeric = np.arange(len(x))  # Assuming x is sorted and continuous weekly dates

         # Generate cubic splines for smooth lines
         x_new = np.linspace(x_numeric.min(), x_numeric.max(), 300)
         spl = make_interp_spline(x_numeric, y, k=3)
         y_smooth = spl(x_new)

         # Convert numeric values back to the original week-based datetime index for plotting
         x_new_dates = pd.to_datetime(x.min()) + pd.to_timedelta(x_new, unit='W')

         plt.plot(x_new_dates, y_smooth, label=label, color=color)

    def save_forecast_to_csv(self, model_name):

        forecast = self.results[model_name]['forecast']
        fitted = self.results[model_name]['fitted']
        dates = pd.date_range(start=self.df.index[-1], periods=len(forecast) + 1, freq='W')[1:]

          # Combine actual, predicted, and forecasted values
        actual_series = self.df.copy()  # Contains actual values up to the max date
        actual_series.name = 'Actual'

         # Create a DataFrame for the forecasted period with NaN for actual values
        history_df = pd.DataFrame({
        'Date': dates,
        'Actual': [np.nan] * len(forecast),
        'Forecast': forecast
         }).set_index('Date')

       # Combine fitted values (in-sample predictions) and future forecasts
        forecast_df = pd.DataFrame({
        'Date': actual_series.index,
        'Actual': actual_series,
        'Forecast': fitted
        }).set_index('Date')

           # Append the future_df to forecast_df
        forecast_df = pd.concat([history_df, forecast_df])
        forecast_df = forecast_df.sort_index()

            # Get the path from the config file and save the CSV
        output_path = self.config['paths']['results']
        csv_path = os.path.join(output_path, f"{model_name}_history_forecast.csv")

        forecast_df.to_csv(csv_path)
        print(f"Forecast saved to {csv_path}")


    def plot_fit_vs_actual(self, model_name):
        plt.figure(figsize=(12, 6))
        ##plt.plot(self.df.index, self.df, label='Actual')
        ###plt.plot(self.df.index, self.results[model_name]['fitted'], color='red', label='Fitted')
        self.smooth_plot(self.df.index, self.df, label='Actual', color='blue')
        self.smooth_plot(self.df.index, self.results[model_name]['fitted'], color='red', label='Fitted')
        plt.legend()
        plt.title(f'Actual vs Fitted for {model_name}')
        plt.savefig(f"{self.config['paths']['plots']}/{model_name}_fit_vs_actual.png")
        #plt.show()

    def plot_forecast(self, model_name):
        forecast = self.results[model_name]['forecast']
        conf_int = self.results[model_name]['conf_int']

        plt.figure(figsize=(12, 6))
        self.smooth_plot(self.df.index, self.df, label='Historical', color='blue')
        ####plt.plot(self.df.index, self.df, label='Historical')    
        forecast_index = pd.date_range(start=self.df.index[-1], periods=len(forecast) + 1, freq='W')[1:]
        self.smooth_plot(forecast_index, forecast, label='Forecast', color='green')
        ####plt.plot(forecast_index, forecast, label='Forecast', color='green')
        plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='green', alpha=0.3)
        plt.legend()
        plt.title(f'Forecast for 80 weeks with 95% CI - {model_name}')
        plt.savefig(f"{self.config['paths']['plots']}/{model_name}_forecast.png")
        plt.legend()
        plt.title(f'Forecast for 80 weeks with 95% CI - {model_name}')
        plt.savefig(f"{self.config['paths']['plots']}/{model_name}_forecast.png")
        #plt.show()


    def plot_residuals(self, model_name):

        residuals = self.results[model_name].get('residuals')

        if residuals is None:
            print(f"No residuals found for model {model_name}")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, residuals, color='blue', label='Residuals')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.legend()
        plt.title(f'Residuals for {model_name}')
        plt.savefig(f"{self.config['paths']['plots']}/{model_name}_residuals.png")
        #plt.show()


    def summarize_results(self):
        summary_df = pd.DataFrame.from_dict(self.results, orient='index', columns=['mse', 'mae'])
        summary_df.to_csv(f"{self.config['paths']['reports']}/model_summary.csv")
        print("Summary of model evaluations saved.")