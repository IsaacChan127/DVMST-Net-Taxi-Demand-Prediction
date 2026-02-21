
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Reshape, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from local_seq_conv import LocalSeqConv
import matplotlib.pyplot as plt

# --- Load pivot data ---
pivot = pd.read_csv("C:/Users/capis/Downloads/pivot_matrix.csv", index_col=0, parse_dates=True).sort_index()

# --- Filter active zones ---
demand_means = pivot.mean()
active_zones = demand_means[demand_means > 1.0].index.astype(str).tolist()

# --- Temporal features ---
pivot['hour'] = pivot.index.hour
pivot['dayofweek'] = pivot.index.dayofweek
hour_oh = pd.get_dummies(pivot['hour'], prefix='hour')
dow_oh = pd.get_dummies(pivot['dayofweek'], prefix='dow')

# --- Training one zone only ---
zone_data = {}
zone = '132'  # Change to any active zone
print(f"\n🚕 Training for Zone {zone}")

# Extract demand
zone_series = pivot[zone]
log1p_demand = np.log1p(zone_series)
label_min = log1p_demand.min()
label_max = log1p_demand.max()
label_max = min(label_max, 10.0)  # clip high end
# Build full features
features = pd.concat([pivot[active_zones], hour_oh, dow_oh], axis=1)
features_logged = np.log1p(features)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_logged)

# Build sequences
seq_len = 8
X, Y = [], []
for i in range(len(features_scaled) - seq_len):
    X.append(features_scaled[i:i+seq_len])
    Y.append(log1p_demand[i+seq_len])

X = np.array(X)
Y = np.array(Y)

# Rescale Y to [0, 1]
label_min = Y.min()
label_max = Y.max()
Y_scaled = (Y - label_min) / (label_max - label_min + 1e-8)

# Dummy topo and spatial inputs
topo_input = np.ones((len(X), 1)) * int(zone)
spatial_input = np.zeros((len(X), seq_len, 9, 9, 1))

# Split data
X_train, X_val, Y_train, Y_val, topo_train, topo_val, spatial_train, spatial_val = train_test_split(
    X, Y_scaled, topo_input, spatial_input, test_size=0.1, random_state=42
)

# --- Define model ---
def build_zone_model(seq_len, num_features):
    temporal_input = Input(shape=(seq_len, num_features), name='temporal_input')
    topo_input = Input(shape=(1,), name='topo_input')
    spatial_input = Input(shape=(seq_len, 9, 9, 1), name='spatial_input')

    spatial = LocalSeqConv(output_dim=32, seq_len=seq_len, kernel_size=(3, 3))(spatial_input)
    spatial = Flatten()(spatial)
    spatial = Reshape((seq_len, -1))(spatial)

    lstm_temporal = LSTM(64)(temporal_input)
    lstm_spatial = LSTM(64)(spatial)
    topo_emb = Dense(16, activation='relu')(topo_input)

    x = Concatenate()([lstm_temporal, lstm_spatial, topo_emb])
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[temporal_input, topo_input, spatial_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mape')
    return model

# --- Train model ---
model = build_zone_model(seq_len=seq_len, num_features=X.shape[2])
model.fit([X_train, topo_train, spatial_train], Y_train,
          validation_data=([X_val, topo_val, spatial_val], Y_val),
          epochs=30, batch_size=64, verbose=2)

# --- Save for evaluation ---
zone_data[zone] = {
    "model": model,
    "scaler": scaler,
    "X_val": X_val,
    "Y_val": Y_val,
    "label_min": label_min,
    "label_max": label_max,
    "spatial_val": spatial_val
}  

# --- Evaluation function ---
def evaluate_zone_model(zone, zone_data):
    data = zone_data[zone]
    model = data["model"]
    pred_scaled = model.predict([
        data["X_val"],
        np.ones((len(data["X_val"]), 1)) * int(zone),
        data["spatial_val"]
    ])[:, 0]

    pred_logged = pred_scaled * (data["label_max"] - data["label_min"]) + data["label_min"]
    actual_logged = data["Y_val"] * (data["label_max"] - data["label_min"]) + data["label_min"]

    pred_demand = np.expm1(pred_logged)
    actual_demand = np.expm1(actual_logged)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(actual_demand, label="Actual", alpha=0.7)
    plt.plot(pred_demand, label="Predicted", alpha=0.7)
    plt.title(f"Zone {zone} — Predicted vs Actual Demand")
    plt.xlabel("Time Step")
    plt.ylabel("Taxi Demand")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return actual_demand, pred_demand
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def print_evaluation_metrics(actual, predicted, zone_id=None):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Avoid division by zero for MAPE
    mask = actual > 0
    if np.any(mask):
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = float('nan')
    
    print(f"\n{'='*40}")
    print(f"Zone {zone_id} Evaluation" if zone_id else "Evaluation Results")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"{'='*40}\n")


# --- Evaluate
actual_demand, pred_demand = evaluate_zone_model(zone, zone_data)
print_evaluation_metrics(actual_demand, pred_demand, zone_id='132')
