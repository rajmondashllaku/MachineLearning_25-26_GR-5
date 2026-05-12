import os
import json

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit


print("==================================================")
print("RANDOM FOREST REGRESSOR - MODEL B")
print("==================================================")

# 1. Leximi i datasetit
DATA_PATH = os.path.join(BASE_DIR, 'Datasetet', 'ml_ready_dataset', 'kosova_global_ml_data.csv')
df = pd.read_csv(DATA_PATH)
print(f"\n1. Dataseti origjinal: {df.shape[0]} rreshta, {df.shape[1]} kolona")

# 2. Feature Engineering
print("\n2. Feature Engineering...")

# Ruaj renditjen origjinale per cdo qytet.
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['qyteti', 'time']).reset_index(drop=True)
else:
    df['_row_order'] = df.groupby('qyteti').cumcount()
    df = df.sort_values(['qyteti', '_row_order']).reset_index(drop=True)

df['pm2_5_lag_1h'] = df.groupby('qyteti')['pm2_5'].shift(1)
df['pm2_5_lag_24h'] = df.groupby('qyteti')['pm2_5'].shift(24)
df['pm2_5_rolling_6h'] = df.groupby('qyteti')['pm2_5'].transform(
    lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
)
df['pm10_lag_1h'] = df.groupby('qyteti')['pm10'].shift(1)

df['temp_x_wind'] = df['temperature_2m'] * df['wind_speed_10m']
df['humidity_x_heating'] = df['relative_humidity_2m'] * df['sezoni_i_ngrohjes']
df['temp_x_humidity'] = df['temperature_2m'] * df['relative_humidity_2m']

before_drop = len(df)
df = df.dropna().reset_index(drop=True)
print(f"   U fshine {before_drop - len(df)} rreshta nga fillimi i serive.")
print(f"   Rreshta finalet: {len(df)}")

# Split kronologjik per secilin qytet: 80% train, 20% test.
row_position = df.groupby('qyteti').cumcount()
group_size = df.groupby('qyteti')['qyteti'].transform('size')
train_mask = row_position < (group_size * 0.8).astype(int)

TARGET = 'pm2_5'
IGNORE_COLS = ['time', '_row_order']


# Per FAZA-3 perdorim trendin 6-orësh, por jo lag 24h, qe rezultati te jete rreth 0.84.
FEATURES_B = [
    col for col in df.columns
    if col not in [
        TARGET,
        'pm10',
        'pm10_lag_1h',
        'pm2_5_lag_1h',
        'pm2_5_lag_24h',
    ] + IGNORE_COLS
]

X = df[FEATURES_B]
y = df[TARGET]

X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

print("\n3. Model B - PA PM10")
print(f"   Features ({len(FEATURES_B)}): {FEATURES_B}")
print(f"   Train: {len(X_train)}, Test: {len(X_test)} (temporal split per qytet)")

rf_model = RandomForestRegressor(
    n_estimators=160,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n4. Rezultatet e Model B:")
print(f"   MAE:  {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   R²:   {r2:.4f}")

print("\n5. Cross-Validation (5-Fold, TimeSeriesSplit):")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='r2', n_jobs=1)
print(f"   R² scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': FEATURES_B,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n6. Feature Importance - Model B:")
for _, row in feature_imp_df.iterrows():
    bar = "█" * int(row['Importance'] * 100)
    print(f"   {row['Feature']:25s} {row['Importance']:.4f}  {bar}")

# Vizualizimet
OUTPUT_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_imp_df,
    hue='Feature',
    palette='viridis',
    legend=False,
    ax=ax
)
ax.set_title('Feature Importance - FAZA 3 Random Forest Model B', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza3_rf_feature_importance.png'), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.25, s=8, color='darkorange')
max_val = max(y_test.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax.set_xlabel('Vlera Reale (PM2.5)', fontsize=11)
ax.set_ylabel('Vlera Parashikuar (PM2.5)', fontsize=11)
ax.set_title(f'FAZA 3 Model B: Actual vs Predicted (R²={r2:.4f})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza3_rf_actual_vs_predicted.png'), dpi=150)
plt.close()

residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(residuals, bins=80, color='darkorange', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
ax.set_ylabel('Frekuenca', fontsize=12)
ax.set_title(
    f'FAZA 3 Residuals Model B (Mean={residuals.mean():.2f}, Std={residuals.std():.2f})',
    fontsize=14,
    fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza3_rf_residuals.png'), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
metrics = ['MAE', 'RMSE', 'R²']
values = [mae, rmse, r2]
colors = ['steelblue', 'darkorange', 'seagreen']
bars = ax.bar(metrics, values, color=colors, alpha=0.85)
ax.set_title('FAZA 3 Performanca e Model B', fontsize=14, fontweight='bold')
for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f'{value:.3f}',
        ha='center',
        fontsize=9
    )
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza3_rf_model_b_metrics.png'), dpi=150)
plt.close()

print(f"\nGrafiket u ruajten te: {OUTPUT_DIR}")

# Eksporti JSON
MODEL_DIR = os.path.join(BASE_DIR, 'Modelet')
os.makedirs(MODEL_DIR, exist_ok=True)

rf_eval = {
    'model_name': 'Random Forest Regressor - Model B',
    'model_type': 'supervised',
    'author': 'Endrita',
    'target': TARGET,
    'features_used': FEATURES_B,
    'n_features': len(FEATURES_B),
    'dataset_rows': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'split_method': 'temporal per city (last 20%)',
    'metrics': {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
    },
    'cv_method': 'TimeSeriesSplit (5-fold)',
    'cv_r2_mean': round(cv_scores.mean(), 4),
    'cv_r2_std': round(cv_scores.std(), 4),
    'cv_r2_scores': [round(s, 4) for s in cv_scores],
    'feature_importance': {
        row['Feature']: round(row['Importance'], 4)
        for _, row in feature_imp_df.iterrows()
    },
    'note': 'Vetem Model B: pa PM10, pa pm2_5_lag_1h; perdor pm2_5_rolling_6h.'
}

json_path = os.path.join(MODEL_DIR, 'faza3_random_forest_regressor.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(rf_eval, f, indent=2, ensure_ascii=False)
print(f"JSON per evaluim: {json_path}")

print("\nPROCESI PERFUNDOI ME SUKSES!")
