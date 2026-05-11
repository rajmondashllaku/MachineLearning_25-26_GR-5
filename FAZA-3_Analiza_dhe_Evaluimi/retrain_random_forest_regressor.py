import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib
import json

# Random Forest Regressor - Parashikimi i PM2.5
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("==================================================")
print("RANDOM FOREST REGRESSOR - PM2.5 PREDICTION")
print("==================================================")

# 1. Leximi i datasetit
DATA_PATH = os.path.join(BASE_DIR, 'Datasetet', 'ml_ready_dataset', 'kosova_global_ml_data.csv')
df = pd.read_csv(DATA_PATH)
print(f"\n1. Dataseti origjinal: {df.shape[0]} rreshta, {df.shape[1]} kolona")

# 2. Feature Engineering - Advanced
print("\n2. Feature Engineering i avancuar...")

# Renditja e saktë Kronologjike!
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['qyteti', 'time']).reset_index(drop=True)

# Lags dhe Rolling Averages
df['pm2_5_lag_1h'] = df.groupby('qyteti')['pm2_5'].shift(1)
df['pm10_lag_1h'] = df.groupby('qyteti')['pm10'].shift(1)

# Mesatarja e 6 orëve të fundit
df['pm2_5_rolling_6h'] = df.groupby('qyteti')['pm2_5'].transform(lambda x: x.shift(1).rolling(window=6, min_periods=1).mean())
df['pm2_5_lag_24h'] = df.groupby('qyteti')['pm2_5'].shift(24)

# Interaction features
df['temp_x_wind'] = df['temperature_2m'] * df['wind_speed_10m']
df['humidity_x_heating'] = df['relative_humidity_2m'] * df['sezoni_i_ngrohjes']
df['temp_x_humidity'] = df['temperature_2m'] * df['relative_humidity_2m']

before_drop = len(df)
df = df.dropna().reset_index(drop=True)
print(f"   U fshinë {before_drop - len(df)} rreshta nga fillimi i serive.")

# ==========================================
# MODEL A: Me PM10 dhe Lag 1h (Baseline me Leakage)
# ==========================================
print("\nMODEL A: Me PM10 + Lag 1h (Lazy Model/Leakage)")

TARGET = 'pm2_5'
KOLONAT_PER_INJORIM = ['time', 'qyteti']

FEATURES_A = [col for col in df.columns if col not in [TARGET] + KOLONAT_PER_INJORIM]
X_a = df[FEATURES_A]
y = df[TARGET]

split_idx = int(len(X_a) * 0.8)
X_train_a, X_test_a = X_a.iloc[:split_idx], X_a.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

rf_a = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_a.fit(X_train_a, y_train)
y_pred_a = rf_a.predict(X_test_a)

mae_a = mean_absolute_error(y_test, y_pred_a)
rmse_a = np.sqrt(mean_squared_error(y_test, y_pred_a))
r2_a = r2_score(y_test, y_pred_a)

print(f"   MAE:  {mae_a:.4f} | RMSE: {rmse_a:.4f} | R²: {r2_a:.4f}")

# ==========================================
# MODEL B: PA PM10 dhe PA LAG_1H (Modeli i vërtetë)
# ==========================================
print("\nMODEL B: Modeli Shkencor (Moti + Trendi 6h)")

FEATURES_B = [col for col in df.columns if col not in [TARGET, 'pm10', 'pm10_lag_1h', 'pm2_5_lag_1h'] + KOLONAT_PER_INJORIM]
X_b = df[FEATURES_B]
X_train_b, X_test_b = X_b.iloc[:split_idx], X_b.iloc[split_idx:]

rf_b = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_b.fit(X_train_b, y_train)
y_pred_b = rf_b.predict(X_test_b)

mae_b = mean_absolute_error(y_test, y_pred_b)
rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_b))
r2_b = r2_score(y_test, y_pred_b)

print(f"   MAE:  {mae_b:.4f} | RMSE: {rmse_b:.4f} | R²: {r2_b:.4f}")

print("\n   Cross-Validation (5-Fold, TimeSeriesSplit):")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(rf_b, X_b, y, cv=tscv, scoring='r2', n_jobs=-1)
print(f"   Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==========================================
# VIZUALIZIMET
# ==========================================
OUTPUT_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Feature Importance
importances_b = rf_b.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': FEATURES_B, 'Importance': importances_b}).sort_values('Importance', ascending=False)
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette="viridis", ax=ax)
ax.set_title('Rëndësia e Faktorëve - Model B (Moti dhe Trendi)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza3_rf_feature_importance.png'), dpi=150)
plt.close()

# 2. Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].scatter(y_test, y_pred_a, alpha=0.2, s=8, color='steelblue')
max_val = max(y_test.max(), y_pred_a.max(), y_pred_b.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[0].set_title(f'Model A: Me Leakage (R²={r2_a:.4f})', fontsize=13, fontweight='bold')
axes[1].scatter(y_test, y_pred_b, alpha=0.2, s=8, color='darkorange')
axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[1].set_title(f'Model B: Korrekt (R²={r2_b:.4f})', fontsize=13, fontweight='bold')
plt.suptitle('Saktësia e Random Forest: Actual vs Predicted', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza3_rf_actual_vs_predicted.png'), dpi=150)
plt.close()

# 3. Residuals (Rikthyer!)
residuals_b = y_test - y_pred_b
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(residuals_b, bins=80, color='darkorange', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
ax.set_ylabel('Frekuenca', fontsize=12)
ax.set_title(f'Residuals Model B (Mean={residuals_b.mean():.2f}, Std={residuals_b.std():.2f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza3_rf_residuals.png'), dpi=150)
plt.close()

# 4. Bar Chart Krahasimi (Rikthyer!)
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['MAE', 'RMSE', 'R²']
vals_a = [mae_a, rmse_a, r2_a]
vals_b = [mae_b, rmse_b, r2_b]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, vals_a, w, label='Model A (me Leakage)', color='steelblue', alpha=0.8)
ax.bar(x + w/2, vals_b, w, label='Model B (Korrekt)', color='darkorange', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title('Krahasimi i Modeleve A vs B', fontsize=14, fontweight='bold')
ax.legend()
for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
    ax.text(i - w/2, va + 0.02, f'{va:.3f}', ha='center', fontsize=9)
    ax.text(i + w/2, vb + 0.02, f'{vb:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza3_rf_model_comparison.png'), dpi=150)
plt.close()

# ==========================================
# RUAJTJA
# ==========================================
MODEL_DIR = os.path.join(BASE_DIR, 'Modelet')
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_b, os.path.join(MODEL_DIR, 'random_forest_pm25.joblib'))

# Ruajtja e rezultateve në txt (Rikthyer!)
results_path = os.path.join(MODEL_DIR, 'rezultatet_random_forest.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write('RANDOM FOREST REGRESSOR - REZULTATET\n')
    f.write('=' * 50 + '\n\n')
    f.write(f'MODEL A (me Leakage):\n  MAE: {mae_a:.4f} | RMSE: {rmse_a:.4f} | R²: {r2_a:.4f}\n\n')
    f.write(f'MODEL B (Modeli Kryesor):\n  MAE: {mae_b:.4f} | RMSE: {rmse_b:.4f} | R²: {r2_b:.4f}\n\n')

# Eksporti JSON
rf_eval = {
    'model_name': 'Random Forest Regressor',
    'model_type': 'supervised',
    'author': 'Endrita',
    'target': 'pm2_5',
    'features_used': FEATURES_B,
    'metrics': {'MAE': round(mae_b, 4), 'RMSE': round(rmse_b, 4), 'R2': round(r2_b, 4)},
    'cv_r2_mean': round(cv_scores.mean(), 4)
}
with open(os.path.join(MODEL_DIR, 'random_forest_regressor.json'), 'w', encoding='utf-8') as f:
    json.dump(rf_eval, f, indent=2)

print("\n[!] PROCESI PËRFUNDOI ME SUKSES! Të gjithë grafikët dhe skedarët u ruajtën.")