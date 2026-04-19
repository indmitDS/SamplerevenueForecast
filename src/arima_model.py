import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    def __init__(self, data):
        self.data = data
        self.model = None

    def select_order(self, max_runs=20):
        # Using auto_arima to select the best (p, d, q) order
        best_aic = float("inf")
        best_order = None
        runs = 0

        auto_arima = pm.auto_arima(
            self.data,
            seasonal=False,  # No seasonal components in ARIMA
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            max_order=20,
            maxiter=100,
            n_jobs=-1
        )

        return auto_arima.order

    def fit(self):
        p, d, q = self.select_order()
        self.model = ARIMA(self.data, order=(p, d, q)).fit()

    def predict(self, steps=80):
        forecast = self.model.get_forecast(steps=steps)
        return forecast.predicted_mean, forecast.conf_int()