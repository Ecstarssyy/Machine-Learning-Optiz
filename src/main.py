# ========== IMPORT LIBRARY ==========
#100 EPOCHS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ========== LOAD DAN EKSPLORASI DATA ==========
df = pd.read_csv('Fashion_Retail_Sales.csv')
df.head()
df.info()
df.describe(include="all")

# ========== DATA CLEANING ==========
# Ganti missing value dengan polarisasi linear
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])
df = df.sort_values('Date Purchase')

# Interpolasi linear untuk missing value
df['Purchase Amount (USD)'] = df['Purchase Amount (USD)'].interpolate(method='linear')
df['Review Rating'] = df['Review Rating'].interpolate(method='linear')

# Pastikan tidak ada missing value tersisa
df = df[df['Purchase Amount (USD)'].notna()]

# #tampilin shorting value
print(df.head())
print(df.tail())

# ========== VISUALISASI DATA SEBELUM EKSTRAKSI ==========
plt.figure(figsize=(15,6))
plt.plot(df['Date Purchase'], df['Purchase Amount (USD)'], label='Original Purchase Amount (USD)', color='orange')
plt.title('Original Purchase Amount (USD) per Transaction', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(pd.Timestamp('2022-10-02'), pd.Timestamp('2023-10-01'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== PREPROCESSING DATA ==========
df_daily = df.set_index('Date Purchase').resample('D').agg({
    'Purchase Amount (USD)': 'sum',
    'Item Purchased': 'count',
    'Review Rating': 'mean'
}).fillna(0)

# Smoothing
df_daily['Purchase Amount (USD)'] = df_daily['Purchase Amount (USD)'].rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')

# Time-based features
df_daily['day_of_week'] = df_daily.index.dayofweek
df_daily['month'] = df_daily.index.month
df_daily['quarter'] = df_daily.index.quarter
df_daily['is_weekend'] = df_daily['day_of_week'].isin([5,6]).astype(int)
df_daily['MA7'] = df_daily['Purchase Amount (USD)'].rolling(window=7).mean()
df_daily['MA30'] = df_daily['Purchase Amount (USD)'].rolling(window=30).mean()
df_daily = df_daily.fillna(method='bfill').fillna(method='ffill')

exog_columns = ['Item Purchased', 'Review Rating', 'day_of_week', 'month', 'quarter', 'is_weekend', 'MA7', 'MA30']

# Scaling exogenous features (penting untuk SARIMAX)
exog_scaler = MinMaxScaler()
df_daily[exog_columns] = exog_scaler.fit_transform(df_daily[exog_columns])

# Scaling target
scaler = MinMaxScaler()
scaled_purchase = scaler.fit_transform(df_daily[['Purchase Amount (USD)']])
df_daily['Purchase Amount (USD)'] = scaled_purchase

# ========== VISUALISASI DATA SESUDAH EKSTRAKSI ==========
plt.figure(figsize=(15,6))
plt.plot(df_daily.index, df_daily['Purchase Amount (USD)'], label='Processed Purchase Amount (USD)', color='blue')
plt.title('Processed Purchase Amount (USD) per Day (Scaled)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD) [Scaled]')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(pd.Timestamp('2022-10-02'), pd.Timestamp('2023-10-01'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== TRAIN-TEST SPLIT (80/20) ==========
n = len(df_daily)
train_size = int(n * 0.8)
test_size = n - train_size

train_data = df_daily.iloc[:train_size]
test_data = df_daily.iloc[train_size:]

train_exog = train_data[exog_columns]
test_exog = test_data[exog_columns]

# ========== SARIMAX PARAMETER TUNING & TRAINING ==========
print("Training Enhanced SARIMAX Model with Exogenous Variables...")
param_combinations = [
    ((1,1,1), (1,1,1,7)),
    ((2,1,2), (1,1,1,7)),
    ((1,1,2), (0,1,1,7)),
    ((2,0,2), (1,1,1,7)),
    ((1,1,1), (2,1,1,7)),
]
best_aic = float('inf')
best_sarimax_model = None
best_params = None

for order, seasonal_order in param_combinations:
    try:
        model = SARIMAX(
            train_data['Purchase Amount (USD)'],
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend='c'
        )
        result = model.fit(disp=False)
        if result.aic < best_aic:
            best_aic = result.aic
            best_sarimax_model = result
            best_params = (order, seasonal_order)
    except Exception as e:
        continue

if best_sarimax_model is None:
    print("Fallback SARIMAX used.")
    model = SARIMAX(
        train_data['Purchase Amount (USD)'],
        exog=train_exog,
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        trend='c'
    )
    best_sarimax_model = model.fit(disp=False)
    best_params = ((1,1,1), (1,1,1,7))

print(f"Best SARIMAX params: {best_params}, AIC: {best_sarimax_model.aic:.2f}")
print("SARIMAX exogenous variables:", exog_columns)

# ========== SARIMAX PREDICTION ==========
forecast = best_sarimax_model.get_forecast(steps=len(test_data), exog=test_exog)
sarimax_pred = forecast.predicted_mean
sarimax_pred_lower = forecast.conf_int().iloc[:, 0]
sarimax_pred_upper = forecast.conf_int().iloc[:, 1]

# Inverse scaling
sarimax_pred = scaler.inverse_transform(sarimax_pred.values.reshape(-1,1)).flatten()
sarimax_pred_lower = scaler.inverse_transform(sarimax_pred_lower.values.reshape(-1,1)).flatten()
sarimax_pred_upper = scaler.inverse_transform(sarimax_pred_upper.values.reshape(-1,1)).flatten()
actual_test = scaler.inverse_transform(test_data['Purchase Amount (USD)'].values.reshape(-1,1)).flatten()

# ========== PERSIAPAN DATA UNTUK GRU ==========
# Untuk GRU: Gabungkan target dan exog sebagai input
gru_features = ['Purchase Amount (USD)'] + exog_columns
gru_data = df_daily[gru_features].values

# Sequence length untuk GRU
n_steps = 14
print(f"Using sequence length of {n_steps} for GRU")

# Create sequences
X, y = [], []
for i in range(n_steps, len(gru_data)):
    X.append(gru_data[i-n_steps:i])
    y.append(gru_data[i, 0])  # target adalah kolom pertama
X, y = np.array(X), np.array(y)

# Split 80/20
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"GRU training data: {X_train.shape}")
print(f"GRU testing data: {X_test.shape}")

# ========== MEMBANGUN MODEL GRU ==========
print("\nBuilding GRU model...")
gru_model = Sequential([
    # Increase complexity and add regularization
    GRU(128, return_sequences=True, input_shape=(n_steps, X.shape[2])),
    GRU(64, return_sequences=True),
    GRU(32, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # Change to linear for regression
])

# Use a lower learning rate
optimizer = Adam(learning_rate=0.0005)
gru_model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])  # Huber loss is more robust
gru_model.summary()

# Callback functions for GRU
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,  # Increase patience
    restore_best_weights=True,
    min_delta=0.00001,  # More sensitive
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,  # More aggressive LR reduction
    patience=15,
    min_lr=1e-7,
    verbose=1
)

# ========== TRAINING MODEL GRU ==========
print("\nTraining GRU model...")
history = gru_model.fit(
    X_train, y_train,
    epochs=300,  # Increase epochs
    batch_size=32,  # Larger batch size
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ========== PREDIKSI DAN INVERSE TRANSFORM GRU ==========
y_pred_gru = gru_model.predict(X_test)
y_pred_gru_inv = scaler.inverse_transform(y_pred_gru)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# ========== EVALUASI ==========
mae_sarimax = mean_absolute_error(actual_test, sarimax_pred)
mape_sarimax = mean_absolute_percentage_error(actual_test, sarimax_pred) * 100
r2_sarimax = r2_score(actual_test, sarimax_pred)
rmse_sarimax = np.sqrt(mean_squared_error(actual_test, sarimax_pred))

mae_gru = mean_absolute_error(y_test_inv, y_pred_gru_inv)
mape_gru = mean_absolute_percentage_error(y_test_inv, y_pred_gru_inv) * 100
r2_gru = r2_score(y_test_inv, y_pred_gru_inv)
rmse_gru = np.sqrt(mean_squared_error(y_test_inv, y_pred_gru_inv))

print("=== Enhanced SARIMAX Evaluation ===")
print(f"RMSE  : {rmse_sarimax:.2f}")
print(f"MAE   : {mae_sarimax:.2f}")
print(f"MAPE  : {mape_sarimax:.2f}%")
print(f"R2    : {r2_sarimax:.4f}")
print("\n=== Enhanced GRU Evaluation ===")
print(f"RMSE  : {rmse_gru:.2f}")
print(f"MAE   : {mae_gru:.2f}")
print(f"MAPE  : {mape_gru:.2f}%")
print(f"R2    : {r2_gru:.4f}")

# ========== GRU 7 DAY FUTURE PREDICTIONS ==========
print("\nPredicting next 7 days with GRU...")
last_sequence = X_test[-1:]
future_predictions_gru = []

for _ in range(7):
    # Get prediction for next day
    next_pred = gru_model.predict(last_sequence)
    future_predictions_gru.append(next_pred[0, 0])
    
    # Update the sequence for next prediction
    new_seq = last_sequence[0][1:].copy()
    new_row = last_sequence[0][-1].copy()
    new_row[0] = next_pred[0, 0]  # Update only the target value
    last_sequence = np.append(new_seq, [new_row], axis=0).reshape(1, n_steps, X.shape[2])

# Inverse transform predictions
future_pred_inv_gru = scaler.inverse_transform(np.array(future_predictions_gru).reshape(-1, 1))

# Create future dates for GRU
future_dates_gru = pd.date_range(start=test_data.index[-1], periods=8)[1:]

# ========== SARIMAX 7 DAY FUTURE PREDICTIONS ==========
print("\nPredicting next 7 days with SARIMAX...")
# Persiapkan data untuk prediksi SARIMAX
last_date = test_data.index[-1]
future_dates_sarimax = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
future_exog = pd.DataFrame(index=future_dates_sarimax, columns=exog_columns)

# Copy pola dari minggu terakhir untuk exogenous features
for col in exog_columns:
    future_exog[col] = df_daily[col].iloc[-7:].values

# Prediksi SARIMAX
future_sarimax = best_sarimax_model.get_forecast(steps=7, exog=future_exog)
future_pred_sarimax = scaler.inverse_transform(future_sarimax.predicted_mean.values.reshape(-1,1)).flatten()

# ========== VISUALISASI HASIL PREDIKSI ==========
# Plot 1: SARIMAX Prediction
plt.figure(figsize=(15,7))
plt.plot(test_data.index, actual_test, label='Actual', linewidth=2, color='black')
plt.plot(test_data.index, sarimax_pred, label='SARIMAX Prediction', linewidth=2, color='blue', alpha=0.8)
plt.fill_between(test_data.index, sarimax_pred_lower, sarimax_pred_upper, 
                 color='blue', alpha=0.1, label='SARIMAX 95% CI')
plt.title('SARIMAX Model Prediction Results', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.text(0.02, 0.95, f'RMSE: {rmse_sarimax:.2f}\nR²: {r2_sarimax:.4f}', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
plt.xlim(test_data.index.min(), test_data.index.max())
plt.tight_layout()
plt.show()

# Plot 2: GRU Prediction
plt.figure(figsize=(15,7))
# Make sure to align test_data.index with y_test_inv length
aligned_test_dates = test_data.index[-len(y_test_inv):]
plt.plot(aligned_test_dates, y_test_inv, 
         label='Actual', linewidth=2, color='black')
plt.plot(aligned_test_dates, y_pred_gru_inv, 
         label='GRU Prediction', linewidth=2, color='green', alpha=0.8)
plt.title('GRU Model Prediction Results', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.text(0.02, 0.95, f'RMSE: {rmse_gru:.2f}\nR²: {r2_gru:.4f}', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

# Plot 3: Combined Predictions
plt.figure(figsize=(15,7))
plt.plot(aligned_test_dates, y_test_inv, label='Actual', linewidth=2, color='black')
plt.plot(aligned_test_dates, y_pred_gru_inv, label='GRU', linewidth=2, color='green', alpha=0.8)
plt.plot(test_data.index, sarimax_pred, label='SARIMAX', linewidth=2, color='blue', alpha=0.8)
plt.title('Comparison of SARIMAX and GRU Predictions', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Purchase Amount (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
text = f'SARIMAX Metrics:\nRMSE: {rmse_sarimax:.2f}\nR²: {r2_sarimax:.4f}\n\n'
text += f'GRU Metrics:\nRMSE: {rmse_gru:.2f}\nR²: {r2_gru:.4f}'
plt.text(0.02, 0.85, text, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.show()

# ========== VISUALISASI 2: BAR CHART PERFORMA ==========
plt.figure(figsize=(10, 6))
metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R² Score']
sarimax_values = [rmse_sarimax, mae_sarimax, mape_sarimax, r2_sarimax*100]
gru_values = [rmse_gru, mae_gru, mape_gru, r2_gru*100]
x = np.arange(len(metrics))
width = 0.35
bars1 = plt.bar(x - width/2, sarimax_values, width, label='SARIMAX', color='blue', alpha=0.7)
bars2 = plt.bar(x + width/2, gru_values, width, label='GRU', color='green', alpha=0.7)
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    if i == 3:
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                 f'{sarimax_values[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                 f'{gru_values[i]/100:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                 f'{sarimax_values[i]:.1f}', ha='center', va='bottom', fontweight='bold')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                 f'{gru_values[i]:.1f}', ha='center', va='bottom', fontweight='bold')
plt.title('Perbandingan Performa Model SARIMAX vs GRU', fontsize=16, fontweight='bold')
plt.xlabel('Metrik Evaluasi')
plt.ylabel('Nilai')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.figtext(0.5, 0.02,
           'Catatan: Untuk RMSE, MAE, MAPE → semakin kecil semakin baik | Untuk R² → semakin besar semakin baik',
           ha='center', fontsize=10, style='italic')
plt.tight_layout()
plt.show()

# ========== RINGKASAN PERFORMA ==========
print(f"\n{'='*50}")
print("RINGKASAN PERFORMA:")
print(f"{'='*50}")
better_rmse = "SARIMAX" if rmse_sarimax < rmse_gru else "GRU"
better_r2 = "SARIMAX" if r2_sarimax > r2_gru else "GRU"
print(f"Model dengan RMSE terbaik: {better_rmse}")
print(f"Model dengan R² terbaik: {better_r2}")
print(f"Improvement SARIMAX: Using best params {best_params}")
print(f"{'='*50}")

# ========== PREDIKSI 7 HARI KE DEPAN ==========
print("\nPrediksi 7 Hari Ke Depan:")
print("================================")
print("Tanggal            SARIMAX     GRU")
print("----------------------------------------")
for i in range(7):
    date = future_dates_gru[i].strftime('%Y-%m-%d')
    sarimax = future_pred_sarimax[i]
    gru = future_pred_inv_gru[i][0]
    print(f"{date}  {sarimax:9.2f}  {gru:9.2f}")

# ========== COMBINED FUTURE PREDICTIONS VISUALIZATION ==========
plt.figure(figsize=(12, 6))
plt.plot(future_dates_gru, future_pred_inv_gru, 'o-', label='GRU Future Forecast', color='green', linewidth=2)
plt.plot(future_dates_sarimax, future_pred_sarimax, 's-', label='SARIMAX Future Forecast', color='blue', linewidth=2)

# Add value annotations
for i in range(len(future_dates_gru)):
    # GRU annotation
    plt.annotate(f"${future_pred_inv_gru[i][0]:.2f}", 
                xy=(future_dates_gru[i], future_pred_inv_gru[i][0]),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='green')
    
    # SARIMAX annotation
    plt.annotate(f"${future_pred_sarimax[i]:.2f}", 
                xy=(future_dates_sarimax[i], future_pred_sarimax[i]),
                xytext=(0, -15),
                textcoords="offset points",
                ha='center', va='top',
                fontsize=8, fontweight='bold', color='blue')

plt.title('7-Day Forecast Comparison: SARIMAX vs GRU', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Purchase Amount (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========== TAMPILKAN 37 HARI TERAKHIR ==========
print("\nPrediksi 73 Hari Terakhir:")
print("================================")
print("Tanggal            Actual      SARIMAX     GRU")
print("------------------------------------------------")

# Align data lengths to ensure we have the same number of predictions from both models
aligned_test_dates_73 = aligned_test_dates[-73:]
gru_pred_73 = y_pred_gru_inv[-73:]
gru_actual_73 = y_test_inv[-73:]

# Find corresponding SARIMAX predictions for the same dates
sarimax_pred_73 = []
for date in aligned_test_dates_73:
    if date in test_data.index:
        idx = test_data.index.get_loc(date)
        sarimax_pred_73.append(sarimax_pred[idx])
    else:
        sarimax_pred_73.append(np.nan)  # Handle missing dates

for i in range(len(aligned_test_dates_73)):
    date = aligned_test_dates_73[i].strftime('%Y-%m-%d')
    actual = gru_actual_73[i][0]
    sarimax = sarimax_pred_73[i] if i < len(sarimax_pred_73) else np.nan
    gru = gru_pred_73[i][0]
    print(f"{date}  {actual:9.2f}  {sarimax:9.2f}  {gru:9.2f}")

print(f"\nNote: Showing last 73 days of predictions")