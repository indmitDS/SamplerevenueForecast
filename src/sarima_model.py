from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import pmdarima as pm
import pandas as pd
import itertools

class SARIMAModel:
    def __init__(self, data, seasonal_period):
        self.data = data
        self.seasonal_period = seasonal_period
        self.model = None

    def select_order_auto(self):
        auto_sarima = pm.auto_arima(self.data, seasonal=True, 
            m=self.seasonal_period, trace=True,error_action='ignore', 
            suppress_warnings=True,stepwise=True, max_order=5,  
            maxiter=5,n_jobs=-1)
        return auto_sarima.order, auto_sarima.seasonal_order

    def select_order(self, max_runs=9):
        # Define possible ranges for p, d, q, P, D, Q
        p = d = q = P = D = Q = range(0, 3)  

        # list of possible (p,d,q) and (P,D,Q) combinations
        pdq_combinations = list(itertools.product(p, d, q))
        seasonal_combinations = list(itertools.product(P, D, Q))

        best_aic = float("inf")
        best_order = None
        best_seasonal_order = None
        runs = 0

        for pdq in pdq_combinations:
            for seasonal_pdq in seasonal_combinations:
                print('run#:', runs)
                if runs >= max_runs:
                    break
                try:
                    # Use auto_arima to fit the model
                    model = pm.auto_arima(
                        self.data,
                        seasonal=True,
                        m=self.seasonal_period,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        start_p=pdq[0], start_q=pdq[2], max_p=pdq[0]+1, max_q=pdq[2]+1,
                        start_P=seasonal_pdq[0], start_Q=seasonal_pdq[2], max_P=seasonal_pdq[0]+1, max_Q=seasonal_pdq[2]+1,
                        max_order=5,
                        maxiter=5,
                        n_jobs=-1
                    )
                    runs += 1

                    # Check if this model has a better AIC
                    if model.aic() < best_aic:
                        best_aic = model.aic()
                        best_order = model.order
                        best_seasonal_order = model.seasonal_order
                except:
                    print(f"Model fitting failed for {pdq} x {seasonal_pdq}: {e}")
                    continue

        return best_order, best_seasonal_order

    def fit(self):
        (p, d, q), (P, D, Q, m) = self.select_order()
        self.model = SARIMAX(self.data, order=(p, d, q), seasonal_order=(P, D, Q, m)).fit()

    def predict(self, steps=80):
        forecast = self.model.get_forecast(steps=steps)
        return forecast.predicted_mean, forecast.conf_int()

    def plot_fit_vs_actual(self):
        fitted = self.model.fittedvalues
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data, label='Actual')
        plt.plot(self.data.index, fitted, color='red', label='Fitted')
        plt.legend()
        plt.title('Actual vs Fitted')
        plt.savefig(f"{self.config['plots']['folder']}/sarima_fit_vs_actual.png")
        plt.show()

    def plot_forecast(self, forecast, conf_int):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data, label='Historical')
        plt.plot(pd.date_range(start=self.data.index[-1], periods=len(forecast) + 1, freq='W')[1:], forecast, label='Forecast', color='green')
        plt.fill_between(pd.date_range(start=self.data.index[-1], periods=len(forecast) + 1, freq='W')[1:], 
                         conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='green', alpha=0.3)
        plt.legend()
        plt.title('Forecast for 80 weeks with 95% CI')
        plt.savefig(f"{self.config['plots']['folder']}/sarima_forecast.png")
        plt.show()