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
import joblib
import mlflow
import mlflow.tensorflow


def read_csv(input_path):
    df = pd.read_csv(input_path)
    return df

def scale_datasets(df, train_ids, test_ids, scaler_path, features_to_scale):
    df_train = df[df['battery_id'].isin(train_ids)]
    df_test = df[df['battery_id'].isin(test_ids)]
    
    scaler = MinMaxScaler()
    
    # 1. Fit scaler on all training data (only selected features)
    train_features_all = df_train[features_to_scale]
    scaler.fit(train_features_all)
    
    # 2. Scale training data per battery using the fitted scaler
    train_scaled = []
    for battery_id in df_train['battery_id'].unique():
        battery_group = df_train[df_train['battery_id'] == battery_id].copy()
        battery_group[features_to_scale] = scaler.transform(battery_group[features_to_scale])
        train_scaled.append(battery_group)
    concat_train = pd.concat(train_scaled).reset_index(drop=True)
    
    # Save the scaler to a file
    joblib.dump(scaler, scaler_path)
    
    # 3. Scale test data per battery using the same scaler (no fitting)
    test_scaled = []
    for battery_id in df_test['battery_id'].unique():
        battery_group = df_test[df_test['battery_id'] == battery_id].copy()
        battery_group[features_to_scale] = scaler.transform(battery_group[features_to_scale])
        test_scaled.append(battery_group)
    concat_test = pd.concat(test_scaled).reset_index(drop=True)
    
    return concat_train, concat_test

def create_lstm_sequences(df_scaled, features, target_col = 'rul', window_size = 10):
    """Set up the input/output structure of function 
    and initialize two empty lists: X and y

    Args:
        df_scaled (dataframe): Scaled dataframe
        features (col): Feature columns
        target_col (col): Target column 
        window_size (float): Window size of the sequences
    """
    X = [] # to store all input sequences
    y = [] # to store corresponding RUL values
            
    # handle each battery separately to keep sequences clean and avoid leakage.
    for battery_id in df_scaled['battery_id'].unique():
        # Create dataset from the per battery id
        df_battery = df_scaled[df_scaled['battery_id'] == battery_id]
        # Sort the battery data by cycle_index
        df_battery = df_battery.sort_values(by='cycle_index')
        # Extract feature and target values
        feature_array = df_battery[features].to_numpy()
        target_array = df_battery[target_col].to_numpy()
        
        for i in range(0, len(feature_array) - window_size + 1):
            sequence_X = feature_array[i : i + window_size] # (0 -> 19)
            target_y = target_array[i + window_size -1] # 19
            
            X.append(sequence_X)
            y.append(target_y)
            
    X = np.array(X)
    y = np.array(y) 
    return X, y

def simple_model(trainX):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=trainX.shape[-2:]))
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=False))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def simple_build_model(trainX):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=trainX.shape[-2:]))
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=False))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def clipped_output(x):
    return tf.clip_by_value(x, 0.0, 1e10)  # clip below 0 to zero

def two_layer_build_model(trainX):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=trainX.shape[-2:]))
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=False))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(trainX, trainY, testX, testY, log_dir_path):
    model = simple_build_model(trainX)
    log_dir = log_dir_path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor= 0.5,
                                  patience=5,
                                  verbose=1)
    early_stopping = EarlyStopping(monitor='val_mae',
                                   patience=10,
                                   restore_best_weights=True)
    history = model.fit(trainX,
                        trainY,
                        epochs=10000,
                        callbacks= [reduce_lr,tensorboard_callback,early_stopping],
                        batch_size=16,
                        validation_data=(testX, testY),
                        verbose=1)
    print('Training is finished!')
    final_val_mae = model.evaluate(testX, testY, verbose=0)[1]
    print(f"Training is finished! Final validation MAE: {final_val_mae:.4f}")
    return model, history

