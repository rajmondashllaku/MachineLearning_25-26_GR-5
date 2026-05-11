import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Random Forest Regressor - Parashikimi i PM2.5
# Perdor lag features, interaction features, cross-validation
# Dy modele: Model A (me PM10) vs Model B (pa PM10)

print("RANDOM FOREST REGRESSOR - PM2.5 PREDICTION")

# 1. Leximi i datasetit
DATA_PATH = os.path.join(BASE_DIR, 'Datasetet', 'ml_ready_dataset', 'kosova_global_ml_data.csv')
df = pd.read_csv(DATA_PATH)
print(f"\n1. Dataseti origjinal: {df.shape[0]} rreshta, {df.shape[1]} kolona")

# 2. Feature Engineering - Lag features dhe Interaction features
print("\n2. Feature Engineering i avancuar...")

# Ruaj renditjen origjinale per cdo qytet.
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['qyteti', 'time']).reset_index(drop=True)
else:
    df['_row_order'] = df.groupby('qyteti').cumcount()
    df = df.sort_values(['qyteti', '_row_order']).reset_index(drop=True)

# Lag features per city
df['pm2_5_lag_1h'] = df.groupby('qyteti')['pm2_5'].shift(1)
df['pm2_5_lag_24h'] = df.groupby('qyteti')['pm2_5'].shift(24)
df['pm10_lag_1h'] = df.groupby('qyteti')['pm10'].shift(1)

# Interaction features
df['temp_x_wind'] = df['temperature_2m'] * df['wind_speed_10m']
df['humidity_x_heating'] = df['relative_humidity_2m'] * df['sezoni_i_ngrohjes']
df['temp_x_humidity'] = df['temperature_2m'] * df['relative_humidity_2m']

before_drop = len(df)
df = df.dropna().reset_index(drop=True)
print(f"   Lag features: fshire {before_drop - len(df)} rreshta (fillimi i serive)")
print(f"   Rreshta finalet: {len(df)}")
print("   Kolonat e reja: pm2_5_lag_1h, pm2_5_lag_24h, pm10_lag_1h, "
      "temp_x_wind, humidity_x_heating, temp_x_humidity")

# Split kronologjik per secilin qytet: 80% train, 20% test.
row_position = df.groupby('qyteti').cumcount()
group_size = df.groupby('qyteti')['qyteti'].transform('size')
train_mask = row_position < (group_size * 0.8).astype(int)

TARGET = 'pm2_5'
IGNORE_COLS = ['time', '_row_order']
y = df[TARGET]

# MODEL A: Me PM10
print("MODEL A: Me PM10 + Lag + Interactions (Baseline)")

FEATURES_A = [col for col in df.columns if col not in [TARGET] + IGNORE_COLS]
X_a = df[FEATURES_A]

X_train_a, X_test_a = X_a.loc[train_mask], X_a.loc[~train_mask]
y_train_a, y_test_a = y.loc[train_mask], y.loc[~train_mask]

rf_a = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_a.fit(X_train_a, y_train_a)
y_pred_a = rf_a.predict(X_test_a)

mae_a = mean_absolute_error(y_test_a, y_pred_a)
rmse_a = np.sqrt(mean_squared_error(y_test_a, y_pred_a))
r2_a = r2_score(y_test_a, y_pred_a)

print(f"   MAE:  {mae_a:.4f}")
print(f"   RMSE: {rmse_a:.4f}")
print(f"   R²:   {r2_a:.4f}")

# MODEL B: PA PM10
print("MODEL B: PA PM10 (Modeli i vertete, pa data leakage)")

FEATURES_B = [
    col for col in df.columns
    if col not in [
        TARGET,
        'pm10',
        'pm10_lag_1h',
        'pm2_5_lag_1h',
    ] + IGNORE_COLS
]

X_b = df[FEATURES_B]

X_train_b, X_test_b = X_b.loc[train_mask], X_b.loc[~train_mask]
y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

print(f"   Features ({len(FEATURES_B)}): {FEATURES_B}")
print(f"   Train: {len(X_train_b)}, Test: {len(X_test_b)} (temporal split per qytet)")

rf_b = RandomForestRegressor(
    n_estimators=160,
    max_depth=3,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_b.fit(X_train_b, y_train)
y_pred_b = rf_b.predict(X_test_b)

mae_b = mean_absolute_error(y_test, y_pred_b)
rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_b))
r2_b = r2_score(y_test, y_pred_b)

print(f"\n   MAE:  {mae_b:.4f}")
print(f"   RMSE: {rmse_b:.4f}")
print(f"   R²:   {r2_b:.4f}")

# Cross-validation
print("\n   Cross-Validation (5-Fold, TimeSeriesSplit):")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(rf_b, X_b, y, cv=tscv, scoring='r2', n_jobs=1)
print(f"   R² scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Krahasimi
print("KRAHASIMI: Model A vs Model B")
print(f"{'Metrika':<10} {'Model A (me PM10)':<22} {'Model B (pa PM10)':<22} {'Ndryshimi'}")
print(f"{'MAE':<10} {mae_a:<22.4f} {mae_b:<22.4f} {mae_b - mae_a:+.4f}")
print(f"{'RMSE':<10} {rmse_a:<22.4f} {rmse_b:<22.4f} {rmse_b - rmse_a:+.4f}")
print(f"{'R²':<10} {r2_a:<22.4f} {r2_b:<22.4f} {r2_b - r2_a:+.4f}")

# Feature Importance
print("FEATURE IMPORTANCE - Model B (pa PM10)")

importances_b = rf_b.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': FEATURES_B,
    'Importance': importances_b
}).sort_values('Importance', ascending=False)

for _, row in feature_imp_df.iterrows():
    bar = "█" * int(row['Importance'] * 100)
    print(f"   {row['Feature']:25s} {row['Importance']:.4f}  {bar}")

# Grafikët
OUTPUT_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 7))
colors = sns.color_palette("viridis", len(feature_imp_df))
bars = ax.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color=colors)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance - FAZA 2 Random Forest', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, feature_imp_df['Importance']):
    ax.text(
        bar.get_width() + 0.003,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.4f}',
        va='center',
        fontsize=9
    )
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza2_rf_feature_importance.png'), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].scatter(y_test_a, y_pred_a, alpha=0.2, s=8, color='steelblue')
max_val = max(y_test_a.max(), y_test.max(), y_pred_a.max(), y_pred_b.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[0].set_xlabel('Vlera Reale (PM2.5)', fontsize=11)
axes[0].set_ylabel('Vlera Parashikuar (PM2.5)', fontsize=11)
axes[0].set_title(f'Model A: Me PM10 (R²={r2_a:.4f})', fontsize=13, fontweight='bold')

axes[1].scatter(y_test, y_pred_b, alpha=0.2, s=8, color='darkorange')
axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
axes[1].set_xlabel('Vlera Reale (PM2.5)', fontsize=11)
axes[1].set_ylabel('Vlera Parashikuar (PM2.5)', fontsize=11)
axes[1].set_title(f'Model B: PA PM10 (R²={r2_b:.4f})', fontsize=13, fontweight='bold')
plt.suptitle('FAZA 2 Random Forest: Actual vs Predicted', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza2_rf_actual_vs_predicted.png'), dpi=150)
plt.close()

residuals_b = y_test - y_pred_b
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(residuals_b, bins=80, color='darkorange', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
ax.set_ylabel('Frekuenca', fontsize=12)
ax.set_title(
    f'FAZA 2 Residuals Model B (Mean={residuals_b.mean():.2f}, Std={residuals_b.std():.2f})',
    fontsize=14,
    fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza2_rf_residuals.png'), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['MAE', 'RMSE', 'R²']
vals_a = [mae_a, rmse_a, r2_a]
vals_b = [mae_b, rmse_b, r2_b]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w / 2, vals_a, w, label='Model A (me PM10)', color='steelblue', alpha=0.8)
ax.bar(x + w / 2, vals_b, w, label='Model B (pa PM10)', color='darkorange', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title('FAZA 2 Krahasimi i Modeleve A vs B', fontsize=14, fontweight='bold')
ax.legend()
for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
    ax.text(i - w / 2, va + 0.02, f'{va:.3f}', ha='center', fontsize=9)
    ax.text(i + w / 2, vb + 0.02, f'{vb:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'faza2_rf_model_comparison.png'), dpi=150)
plt.close()

print(f"\nGrafiket u ruajten te: {OUTPUT_DIR}")

# Eksporti JSON
MODEL_DIR = os.path.join(BASE_DIR, 'Modelet')
os.makedirs(MODEL_DIR, exist_ok=True)
rf_eval = {
    'model_name': 'Random Forest Regressor',
    'model_type': 'supervised',
    'author': 'Endrita',
    'target': 'pm2_5',
    'features_used': FEATURES_B,
    'n_features': len(FEATURES_B),
    'dataset_rows': len(df),
    'train_size': len(X_train_b),
    'test_size': len(X_test_b),
    'split_method': 'temporal per city (last 20%)',
    'metrics': {
        'MAE': round(mae_b, 4),
        'RMSE': round(rmse_b, 4),
        'R2': round(r2_b, 4),
    },
    'cv_method': 'TimeSeriesSplit (5-fold)',
    'cv_r2_mean': round(cv_scores.mean(), 4),
    'cv_r2_std': round(cv_scores.std(), 4),
    'cv_r2_scores': [round(s, 4) for s in cv_scores],
    'feature_importance': {
        row['Feature']: round(row['Importance'], 4)
        for _, row in feature_imp_df.iterrows()
    },
    'model_a_r2': round(r2_a, 4),
    'note': 'Model B (pa PM10) eshte modeli kryesor. Model A ka data leakage.'
}

json_path = os.path.join(MODEL_DIR, 'random_forest_regressor.json')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(rf_eval, f, indent=2, ensure_ascii=False)

print(f"JSON per evaluim: {json_path}")

print("PERFUNDOI ME SUKSES!")
