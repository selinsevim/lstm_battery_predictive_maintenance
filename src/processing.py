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


def read_csv(input_path):
    df = pd.read_csv(input_path)
    return df

def inspect_data(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isna().any())
    df = df.rename(columns={"Cycle_Index": "cycle_index", "Discharge Time (s)": "discharge_time_s", "Decrement 3.6-3.4V (s)": "decrement_3_6_3_4V",
                   "Max. Voltage Dischar. (V)": "max_voltage_discharge_v", "Min. Voltage Charg. (V)": "min_voltage_charge_v", "Time at 4.15V (s)": "time_at_4_15V_s",
                   "Time constant current (s)": "time_constant_current_s", "Charging time (s)": "charging_time_s", "RUL": "rul"})
    
    # Create battery_id
    df['battery_id']= 0 
    batteries=[] 
    ID=1
    for rul in df['rul']: 
        batteries.append(ID) 
        if rul == 0: 
            ID+=1
            continue
    df['battery_id'] = batteries 
    return df

def rul_chart(df):
    battery_list = [1, 2, 3, 4, 5] 
    for battery_id in battery_list:
        fig = px.line(
            df[df['battery_id'] == battery_id],
            x='cycle_index',
            y='rul',
            title=f'Battery ID: {battery_id} - RUL over Cycles'
        )
        fig.update_traces(line_color='#ee8ef5')
        fig.update_layout(
        width=600,
        height=400,
        font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        )
        )

        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showline=True,            # show axis line
                linecolor='darkgray',        # axis line color
                showgrid=True,            # show grid lines
                gridcolor='lightgray',    # grid line color
                gridwidth=1
            ),
            yaxis=dict(
                showline=True,
                linecolor='darkgray',
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            )
        )
        fig.show()
        time.sleep(5)

def corr_matrix(df):
    df_corr = df.drop(columns='battery_id')
    corr_matrix = df_corr.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Plotly3_r',  
        text=corr_matrix.values,  # Add the correlation values as text
        texttemplate="%{z:.2f}",  # Format the text (2 decimal places)
        showscale=True
    ))
    fig.update_layout(title=f'Correlation Heatmap of Battery ID {battery_id}')
    fig.update_layout(
        width=1000,
        height=800,
        font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        )
    )
    fig.show()
    
def scatter_matrix(df):
    battery_list = [1, 2, 3, 4, 5] 
    for battery_id in battery_list:
        fig = px.scatter_matrix(df[df['battery_id'] == battery_id],
            dimensions=['discharge_time_s', 'decrement_3_6_3_4V',
            'max_voltage_discharge_v', 'min_voltage_charge_v', 'time_at_4_15V_s',
            'time_constant_current_s', 'charging_time_s'],
            color='rul', 
            color_continuous_scale='Plotly3_r'
        )

        fig.update_layout(
            title=dict(text=f'Scatter matrix of Battery ID {battery_id}'),
            width=1200,
            height=1200,
            font=dict(
                family="Arial, Courier New, monospace",  # font family
                size=12,                                 # font size (pixels)
                color="darkblue"                         # font color
            )
        )

        fig.show()
        
def box_plot(df):
    battery_list = [1, 2, 3, 4, 5] 
    for battery_id in battery_list:
        fig = go.Figure()
        fig.add_trace(go.Box(y= df[df['battery_id']==battery_id]["max_voltage_discharge_v"],
                    marker_color = 'indianred', name = 'max_voltage_discharge_v'))
        fig.add_trace(go.Box(y = df[df['battery_id']==battery_id]["min_voltage_charge_v"],
                    marker_color = 'lightseagreen', name= 'min_voltage_charge_v'))

        fig.update_layout(
            title=dict(text=f"Box plot of {battery_id}"),
            width=800,
            height=400
        )
        fig.update_layout(
        font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        )
        )

        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showline=True,            # show axis line
                linecolor='darkgray',        # axis line color
                showgrid=True,            # show grid lines
                gridcolor='lightgray',    # grid line color
                gridwidth=1
            ),
            yaxis=dict(
                showline=True,
                linecolor='darkgray',
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            ))
        fig.show()
        
def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[feature] < lower) | (data[feature] > upper)]

def flag_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)

def clean_battery_data(df, features, method="interpolate"):
    df_cleaned = df.copy()
    
    for battery_id in df['battery_id'].unique():
        battery_df = df_cleaned[df_cleaned['battery_id'] == battery_id].copy()
        
        for feature in features:
            outlier_mask = flag_outliers_iqr(battery_df[feature])
            outlier_indices = battery_df[outlier_mask].index

            if method == "interpolate":
                battery_df.loc[outlier_mask, feature] = np.nan
                battery_df[feature] = battery_df[feature].interpolate(method='linear', limit_direction='both')
                 
            elif method == "clip":
                # Clip to non-outlier range
                Q1 = battery_df[feature].quantile(0.25)
                Q3 = battery_df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                battery_df[feature] = battery_df[feature].clip(lower, upper)
            
            elif method == "drop":
                battery_df = battery_df.drop(index=outlier_indices)

        # Replace cleaned version
        df_cleaned.loc[battery_df.index] = battery_df

    return df_cleaned