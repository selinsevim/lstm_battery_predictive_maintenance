# Battery RUL Predictive Maintenance using LSTM

This project leverages Long Short-Term Memory (LSTM) neural networks to predict the **Remaining Useful Life (RUL)** of batteries based on sequential time-series data. Accurate prediction of RUL helps in proactive maintenance and enhances the safety and efficiency of battery-powered systems.

```bash
          ┌────────────┐
Battery 1 │  Cycles 1–20  ──> X1, y1 (RUL @ 20)
          │  Cycles 2–21  ──> X2, y2 (RUL @ 21)
          │    ...        ──> ...
          └────────────┘

          ┌────────────┐
Battery 2 │  Cycles 1–20  ──> X1, y1
          │  Cycles 2–21  ──> X2, y2
          └────────────┘

          Combine all into:
          X = [X1, X2, ..., Xn]  → shape (n_samples, 20, num_features)
          y = [y1, y2, ..., yn]  → shape (n_samples,)

```

---

## Project Structure

```bash
├── data/                    # Stored data
│   └── raw/
│   └── processed/
├── src/
│   └── inference.py         # Inference file
│   └── model.py             # Training file
│   └── processing.py        # Processing file
├── scripts/
│   └── run_training.py      # Run training file
│   └── run_processing.py    # Run processing file
├── logs/                    # TensorBoard logs
├── analysis/                # Analysis notebook
├── models/                  # Trained models
├── ml_runs                  # Experiments meta data
├── README.md
├── requirements.txt
```

## Project Highlights

- LSTM-based deep learning model for time-series prediction
- Feature scaling and per-battery preprocessing for robust model generalization
- Visual exploration with Plotly (scatter matrices, correlation heatmaps, RUL plots)
- Multiple LSTM architectures (simple, two-layer with dropout)
- TensorBoard logging and callbacks for training monitoring
- MLFlow for experiment saving
- Clean separation of training and testing battery IDs to avoid data leakage

---

## Model Objective

Predict the **Remaining Useful Life (RUL)** of lithium-ion batteries from features collected during their charge/discharge cycles.

---

## Tech Stack

| Category      | Tools Used                                    |
| ------------- | --------------------------------------------- |
| Programming   | Python 3                                      |
| ML Framework  | TensorFlow / Keras                            |
| Data Handling | Pandas, NumPy                                 |
| Visualization | Plotly (Express & Graph Objects)              |
| Scaling       | Scikit-learn (MinMaxScaler)                   |
| Callbacks     | TensorBoard, EarlyStopping, ReduceLROnPlateau |

---

## Features Used

The following features were used per cycle for each battery.These features are scaled using MinMaxScaler per training set and reused for the test set to avoid leakage:

| Category                |
| ----------------------- |
| discharge_time_s        |
| decrement_3_6_3_4V      |
| max_voltage_discharge_v |
| min_voltage_charge_v    |
| time_at_4_15V_s         |
| time_constant_current_s |
| charging_time_s         |

## Data Processing Pipeline

1. Read and Inspect CSV data
2. Rename columns and create battery_id sequences
3. Generate visualizations for correlation and feature exploration
4. Scale features using MinMaxScaler
5. Create LSTM-ready sequences per battery with a rolling window
6. Train/Test Split by battery_id

## How to Run

1. Install requirements (if not already):

```bash
pip install pandas numpy plotly scikit-learn tensorflow joblib
```

2. Update paths and run preprocessing:

```bash
df = read_csv('data/your_dataset.csv')
df = inspect_data(df)
```

3. Scale and prepare sequences:

```bash
train_df, test_df = scale_datasets(df, train_ids=[1,2,3], test_ids=[4,5], scaler_path='scaler.joblib', features_to_scale=[...])
trainX, trainY = create_lstm_sequences(train_df, features=features_list)
testX, testY = create_lstm_sequences(test_df, features=features_list)
```

4. Train the model:

```bash
model, history = train_model(trainX, trainY, testX, testY, log_dir_path='logs/')
```

5. Monitor training in TensorBoard:

```bash
tensorboard --logdir=logs/
```
