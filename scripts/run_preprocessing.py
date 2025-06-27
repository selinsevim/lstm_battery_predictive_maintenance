import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorboard
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
from tensorflow.python import keras
from src.processing import *


input_path = r'/Users/selinsevim/Desktop/predictive_maintenance_rul_battery_model/data/raw/Battery_RUL.csv'
output_path = r'/Users/selinsevim/Desktop/predictive_maintenance_rul_battery_model/data/processed/processed.csv'

df = read_csv(input_path)
inspect_data(df)

df.to_csv(output_path)