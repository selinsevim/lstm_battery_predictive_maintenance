# Battery Predictive Maintenance using LSTM

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

## Project Structure

```bash
├── data/
│   └── your_dataset.csv
├── processing.py            # Data cleaning, feature selection, visualizations
├── model.py                 # Model building, training, and evaluation
├── logs/                    # TensorBoard logs
├── saved_models/            # Trained models
├── scaler.joblib            # Saved MinMaxScaler
├── README.md

```
