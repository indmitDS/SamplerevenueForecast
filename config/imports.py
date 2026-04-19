import os
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import yaml
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from lazypredict.Supervised import LazyRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import LSTM, Dense

import pickle
from datetime import timedelta, datetime

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import itertools

from prophet import Prophet

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'