'''
import os 
####import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from config import *
'''

import os
import logging
import logging.config

# Base paths (ADD THIS BLOCK HERE 👇)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")

LOGGING_CONF = os.path.join(CONFIG_DIR, "logging.conf")
CONFIG_YAML = os.path.join(CONFIG_DIR, "config.yaml")

# env settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import *

def main():
    # Initialize logging
    logging.config.fileConfig(LOGGING_CONF)
    ###logging.config.fileConfig(os.path.join('config', 'logging.conf'))
    logger = logging.getLogger(__name__)
    logger.info("Starting the forecasting pipeline.")
    
    try:
        # Load configuration
        config_path = CONFIG_YAML
        ##config_path = os.path.join('config', 'config.yaml')
        config = load_config(config_path)
        data_preprocessor = DataPreprocessing(config)

        # Load data
        data = data_preprocessor.load_data()

        

        # Feature Engineering and Aggregation
        data_preprocessor.filter_data()  # Filter data based on conditions
        agg_data = data_preprocessor.convert_to_yyyyww()

        # Perform feature analysis
        feature_analysis = FeatureAnalysis(agg_data, config)
        feature_importance = feature_analysis.feature_importance_analysis()
        f_stat, p_value = feature_analysis.perform_anova()
        feature_analysis.plot_customer_no_vs_revenue()
        feature_analysis.plot_revenue_by_customer()

        # Assuming feature analysis showed CUSTOMER_NO is not important
        logger.info("Dropping CUSTOMER_NO and CUSTOMER_NO_ENCODED columns.")
        agg_data = agg_data.drop(columns=['CUSTOMER_NO', 'CUSTOMER_NO_ENCODED'])
        agg_data = agg_data.groupby('YYYYWW').sum().reset_index()

        ## Set Datettime Index
        agg_data['Date'] = pd.to_datetime(agg_data['YYYYWW'] + '-1', format='%Y-%W-%w')
        agg_data.set_index('Date', inplace=True)
        agg_data = agg_data.asfreq('W-MON') 

        # Time Trend
        feature_analysis.plot_revenue_vs_time()

        # Forward Fill
        ###agg_data['TOTAL_REVENUE_USD'].fillna(method='ffill', inplace=True)
        # Spline interpolation assuming non linear trend
        ###agg_data['TOTAL_REVENUE_USD'] = agg_data['TOTAL_REVENUE_USD'].interpolate(method='spline', order=3)

        # Time interpolation 
        agg_data['TOTAL_REVENUE_USD'] = agg_data['TOTAL_REVENUE_USD'].interpolate(method='time')


        agg_data = agg_data.drop('YYYYWW', axis=1)
        df = agg_data['TOTAL_REVENUE_USD']

        df.to_csv('agg_data.csv')####, index=False)

        # Initialize models
        sarima_model = SARIMAModel(df, seasonal_period=52)
        arima_model = ARIMAModel(df)
        prophet_model = ProphetModel(df)
        lstm_model = LSTMModel(df, look_back=52)


        #Fit models
        arima_model.fit()
        prophet_model.fit()
        lstm_model.fit(epochs=90, batch_size=16)
        sarima_model.fit()


         # Evaluatation Instaniation
        evaluation = ModelEvaluation(df, 'config/config.yaml')


        ### ARIMA 

        evaluation.evaluate_model(arima_model, 'ARIMA')
        evaluation.plot_fit_vs_actual('ARIMA')
        evaluation.plot_forecast('ARIMA')
        evaluation.plot_residuals('ARIMA')
        evaluation.save_forecast_to_csv('ARIMA')


        ### Prophet
        evaluation.evaluate_model(prophet_model, 'Prophet')
        evaluation.plot_fit_vs_actual('Prophet')
        evaluation.plot_forecast('Prophet')
        evaluation.plot_residuals('Prophet')
        evaluation.save_forecast_to_csv('Prophet')


        ###  LSTM 
        evaluation.evaluate_model(lstm_model, 'LSTM')
        evaluation.plot_fit_vs_actual('LSTM')
        evaluation.plot_forecast('LSTM')
        evaluation.plot_residuals('LSTM')
        evaluation.save_forecast_to_csv('LSTM')

        ### SARIMA 

        evaluation.evaluate_model(sarima_model, 'SARIMA')
        evaluation.plot_fit_vs_actual('SARIMA')
        evaluation.plot_forecast('SARIMA')
        evaluation.plot_residuals('SARIMA')
        evaluation.save_forecast_to_csv('SARIMA')




    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()