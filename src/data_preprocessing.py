import pandas as pd
import logging
import os
import yaml
from datetime import datetime, timedelta

class DataPreprocessing:
    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.data = None

    def load_data(self):
        try:
            file_path = self.config['data']['input_file']
            sheet_name = self.config['data']['sheet_name']
            self.logger.info(f"Loading data from {file_path}, sheet: {sheet_name}")

            self.data = pd.read_excel(file_path, sheet_name=sheet_name)
            # Drop specified columns in config file
            columns_to_drop = self.config['data'].get('columns_to_drop', [])
            if columns_to_drop:
                self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
                self.logger.info(f"Dropped columns: {columns_to_drop}")

            # Drop duplicates
            self.data.drop_duplicates(inplace=True)
            self.logger.info("Dropped duplicate rows.")

            self.logger.info("Data loaded successfully.")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def filter_data(self):
      self.logger.debug("Filtering data based on conditions.")
      try:
        if self.data is None:
            raise ValueError("Data is not loaded. Please ensure `load_data()` has been called before filtering.")

        for column, value in self.config['data']['filter_conditions'].items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in the data.")
            self.data = self.data[self.data[column] == value]

        self.logger.info("Data filtered successfully.")
        return self.data
      except Exception as e:
        self.logger.error(f"Error filtering data: {e}")
        raise

    def convert_to_yyyyww(self):
      self.logger.debug("Converting fiscal year and fiscal week to YYYYWW format.")
      try:
        # Retrieve column names from config
        fiscal_year_col = self.config['data']['fiscal_year_column']
        fiscal_week_col = self.config['data']['fiscal_week_column']
        customer_col = self.config['data']['customer_column']
        target_col = self.config['data']['target_column']

        # Create the YYYYWW column based on FISCAL_YEAR and FISCAL_WEEK_IN_QUARTER
        self.data['YYYYWW'] = self.data[fiscal_year_col].astype(str) + '-' + self.data[fiscal_week_col].astype(str).str.zfill(2)

        # Group by YYYYWW and CUSTOMER_NO, summing the TOTAL_REVENUE_USD
        grouped_data = self.data.groupby(['YYYYWW', customer_col]).agg({
            target_col: 'sum'
        }).reset_index()

        # Set YYYYWW as the index
        grouped_data.set_index('YYYYWW', inplace=True)


        grouped_data = grouped_data.reset_index()

        return grouped_data
      except Exception as e:
        self.logger.error(f"Error converting to YYYYWW: {e}")
        raise