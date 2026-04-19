import logging
import os
import matplotlib.pyplot as plt

class ExploratoryDataAnalysis:
    def __init__(self, data, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = data
        self.config = config

    def plot_time_series(self):

        self.logger.debug("Plotting time series.")
        try:
            plt.figure(figsize=(14, 7))
            plt.plot(self.data['Date'], self.data[self.config['data']['target_column']], marker='o', linestyle='-', color='b')
            plt.title('Historical Total Revenue (USD)')
            plt.xlabel('Date')
            plt.ylabel('Total Revenue (USD)')
            plt.grid(True)
            plot_path = os.path.join(self.config['paths']['eda'], 'historical_revenue.png')
            plt.savefig(plot_path)
            plt.show()
        except Exception as e:
            self.logger.error(f"Error plotting time series: {e}")
            raise