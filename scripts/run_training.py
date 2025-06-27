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
import os
import sys
from pathlib import Path

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import *
from src.inference import *


mlflow.set_tracking_uri(r"/Users/selinsevim/Desktop/predictive_maintenance_rul_battery_model/ml_runs")
mlflow.set_experiment("batter_predictive_01")

log_dir_path = r'/Users/selinsevim/Desktop/predictive_maintenance_rul_battery_model/logs/fit/'
input_path = r'/Users/selinsevim/Desktop/predictive_maintenance_rul_battery_model/data/processed/processed.csv'
scaler_path = r'/Users/selinsevim/Desktop/predictive_maintenance_rul_battery_model/model/scaler.pkl'
features_to_scale = ['discharge_time_s',
       'max_voltage_discharge_v', 'min_voltage_charge_v', 'time_at_4_15V_s',
       'time_constant_current_s', 'charging_time_s']
features_to_train = ['discharge_time_s',
       'max_voltage_discharge_v', 'min_voltage_charge_v', 'time_at_4_15V_s',
       'time_constant_current_s', 'charging_time_s']

model_path = r'/Users/selinsevim/Desktop/predictive_maintenance_rul_battery_model/model/model.keras'

train_ids = [1,2,3,4,5,6,7,8,9,10,11,12]
test_ids = [13,14]
window_size = 10


df = read_csv(input_path)
df_train_scaled, df_test_scaled = scale_datasets(df, train_ids, test_ids, scaler_path, features_to_scale)

X_train,y_train = create_lstm_sequences(df_scaled=df_train_scaled, features=features_to_train, target_col='rul', window_size=10)
X_test, y_test = create_lstm_sequences(df_scaled=df_test_scaled, features=features_to_train, target_col='rul', window_size=10)


with mlflow.start_run(): 
    # Log parameters
    mlflow.log_param("future_target", 'rul')
    mlflow.log_param("window_size", 10)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", 1000)
    mlflow.log_param("model_type", "simple_model")
    mlflow.log_param("patience", 10)
    mlflow.log_param("features", df_train_scaled.columns.tolist())

    model, history = train_model(X_train, 
                             y_train, 
                             X_test, 
                             y_test,
                             log_dir_path)
    
    # Log metrics
    val_loss = history.history['val_loss'][-1]
    training_loss = history.history['loss'][-1]
    val_mae = history.history['val_mae'][-1]
    print("Training is finished")
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("training_loss", training_loss)
    mlflow.log_metric("val_mae", val_mae)
    
    # Log TensorBoard as an artifact
    mlflow.log_artifacts(log_dir_path, artifact_path="tensorboard_logs")
    print("TensorBoard logs saved to MLflow.")
    
    # Log scaler as artifact
    mlflow.log_artifact(scaler_path)
    
model.save(model_path)

# Run inference
df_inference = inference(X_test, y_test, model)