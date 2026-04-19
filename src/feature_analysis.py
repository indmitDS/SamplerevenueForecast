import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import seaborn as sns
from scipy.stats import f_oneway

class FeatureAnalysis:
    def __init__(self, data, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = data
        self.config = config

    def encode_customer_no(self):
        self.logger.debug("Encoding CUSTOMER_NO feature.")
        try:
            le = LabelEncoder()
            self.data['CUSTOMER_NO_ENCODED'] = le.fit_transform(self.data['CUSTOMER_NO'])
            return self.data
        except Exception as e:
            self.logger.error(f"Error encoding CUSTOMER_NO: {e}")
            raise

    def remove_duplicates_and_drop_yyyyww(self, data):
      self.logger.debug("Dropping YYYYWW column and removing duplicate rows.")
      try:
        # Create a copy of the data to avoid altering self.data
        temp_data = data.copy()

        # Drop the YYYYWW column
        temp_data = temp_data.drop(columns=['YYYYWW'])

        # Remove duplicate rows
        temp_data = temp_data.drop_duplicates()

        self.logger.info("YYYYWW column dropped and duplicates removed.")
        return temp_data
      except Exception as e:
        self.logger.error(f"Error dropping YYYYWW column and removing duplicates: {e}")
        raise

    def feature_importance_analysis(self):
        self.logger.debug("Performing feature importance analysis.")
        try:
            # Ensure CUSTOMER_NO is encoded
            if 'CUSTOMER_NO_ENCODED' not in self.data.columns:
                self.encode_customer_no()

            # Drop the YYYYWW column and remove duplicates on a copy of the data
            analysis_data = self.remove_duplicates_and_drop_yyyyww(self.data)

            # Prepare data for feature importance analysis
            X = analysis_data.drop(columns=[self.config['data']['target_column'], 'CUSTOMER_NO'])
            y = analysis_data[self.config['data']['target_column']]


            # Fit XGBoost model
            model = XGBRegressor(objective='reg:squarederror')
            model.fit(X, y)

            # Get feature importances
            importance = model.feature_importances_
            feature_names = X.columns

            # Plotting the feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, importance)
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Features")

            # Save plot to the specified directory
            plots_dir = self.config['paths']['plots']  # Changed to use 'eda' path as per typical naming conventions
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, 'feature_importance.png')
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Feature importance plot saved to {plot_path}")

            return importance
        except Exception as e:
            self.logger.error(f"Error performing feature importance analysis: {e}")
            raise

    def perform_anova(self):
        self.logger.debug("Performing ANOVA to assess the significance of CUSTOMER_NO.")
        try:
            # Perform one-way ANOVA
            unique_customers = self.data['CUSTOMER_NO'].unique()
            groups = [self.data[self.data['CUSTOMER_NO'] == customer][self.config['data']['target_column']] for customer in unique_customers]

            # Perform ANOVA
            f_stat, p_value = f_oneway(*groups)


            self.logger.info(f"ANOVA results: F-statistic = {f_stat}, p-value = {p_value}")

            if p_value < 0.05:
                self.logger.info("The effect of CUSTOMER_NO on TOTAL_REVENUE_USD is statistically significant.")
            else:
                self.logger.info("The effect of CUSTOMER_NO on TOTAL_REVENUE_USD is not statistically significant.")

            # Save ANOVA results to a file
            results_dir = self.config['paths']['results']
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, 'anova_results.txt')
            with open(results_path, 'w') as file:
                file.write(f"ANOVA Results:\n")
                file.write(f"F-statistic: {f_stat}\n")
                file.write(f"p-value: {p_value}\n")
                if p_value < 0.05:
                    file.write("Conclusion: The effect of CUSTOMER_NO on TOTAL_REVENUE_USD is statistically significant.\n")
                else:
                    file.write("Conclusion: The effect of CUSTOMER_NO on TOTAL_REVENUE_USD is not statistically significant.\n")

            self.logger.info(f"ANOVA results saved to {results_path}")
            return f_stat, p_value
        except Exception as e:
            self.logger.error(f"Error performing ANOVA: {e}")
            raise

    def plot_customer_no_vs_revenue(self):
        self.logger.debug("Plotting CUSTOMER_NO vs TOTAL_REVENUE_USD.")
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='CUSTOMER_NO', y=self.config['data']['target_column'], data=self.data)
            plt.title("Boxplot of TOTAL_REVENUE_USD by CUSTOMER_NO")
            plt.xticks(rotation=90)
            plt.grid(True)

            # Save plot to the specified directory
            plots_dir = self.config['paths']['plots']
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, 'customer_no_vs_revenue.png')
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Customer vs Revenue plot saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting CUSTOMER_NO vs TOTAL_REVENUE_USD: {e}")
            raise

    def plot_revenue_by_customer(self):

        self.logger.debug("Plotting TOTAL_REVENUE_USD as a function of YYYYWW, labeled by CUSTOMER_NO.")

        try:
            plt.figure(figsize=(14, 8))
            for customer in self.data['CUSTOMER_NO'].unique():
                customer_data = self.data[self.data['CUSTOMER_NO'] == customer]
                plt.plot(customer_data['YYYYWW'], customer_data[self.config['data']['target_column']], label=customer)

            plt.title("TOTAL_REVENUE_USD Over Time by CUSTOMER_NO")
            plt.xlabel("YYYYWW")
            plt.ylabel("TOTAL_REVENUE_USD")
            plt.xticks(rotation=45)
            plt.legend(title='CUSTOMER_NO')
            plt.grid(True)

            plots_dir = self.config['paths']['plots']
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, 'revenue_by_customer.png')
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Revenue by Customer plot saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting TOTAL_REVENUE_USD by CUSTOMER_NO: {e}")
            raise

    def plot_revenue_vs_time(self):
      self.logger.debug("Plotting Revenue vs Time (YYYYWW).")
      try:
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.config['data']['target_column']], marker='o', linestyle='-')
        plt.title('Revenue vs Time (YYYYWW)')
        plt.xlabel('Time (YYYYWW)')
        plt.ylabel('Total Revenue (USD)')
        plt.xticks(ticks=self.data.index[::len(self.data)//10], labels=self.data['YYYYWW'].iloc[::len(self.data)//10], rotation=45)
        plt.grid(True)


        plots_dir = self.config['paths']['plots']
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, 'revenue_vs_yyyyww.png')
        plt.savefig(plot_path)
        plt.close()

        self.logger.info(f"Revenue vs Time plot saved to {plot_path}")
      except Exception as e:
        self.logger.error(f"Error plotting Revenue vs Time: {e}")
        raise