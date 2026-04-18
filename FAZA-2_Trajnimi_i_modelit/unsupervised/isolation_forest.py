import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("ISOLATION FOREST - ANOMALY DETECTION")

# 1. Leximi i datasetit
DATA_PATH = os.path.join(BASE_DIR, 'Datasetet', 'ml_ready_dataset', 'kosova_global_ml_data.csv')
df = pd.read_csv(DATA_PATH)
print(f"\n1. Dataseti: {df.shape[0]} rreshta, {df.shape[1]} kolona")

qytetet_emrat = {1: 'Prishtine', 2: 'Prizren', 3: 'Peje'}

# 2. Feature Engineering
print("\n2. Feature Engineering per anomaly detection...")
df['temp_x_wind'] = df['temperature_2m'] * df['wind_speed_10m']
df['humidity_x_heating'] = df['relative_humidity_2m'] * df['sezoni_i_ngrohjes']
df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 0.01)  # PM2.5/PM10 ratio (anomalous when very high)

FEATURES = ['pm10', 'pm2_5', 'temperature_2m', 'relative_humidity_2m',
            'surface_pressure', 'wind_speed_10m', 'temp_x_wind',
            'humidity_x_heating', 'pm_ratio']

print(f"   Features: {FEATURES}")

# MODEL A: Global (V1 baseline per krahasim)
print("MODEL A: Global Isolation Forest (baseline)")

scaler_global = StandardScaler()
X_global = scaler_global.fit_transform(df[FEATURES])

iso_global = IsolationForest(
    n_estimators=200, contamination=0.05,
    max_samples='auto', random_state=42, n_jobs=-1
)
iso_global.fit(X_global)

df['score_global'] = iso_global.decision_function(X_global)
df['anomaly_global'] = iso_global.predict(X_global)

n_anom_global = (df['anomaly_global'] == -1).sum()
print(f"   Anomali globale: {n_anom_global} ({n_anom_global/len(df)*100:.1f}%)")

for kodi, emri in qytetet_emrat.items():
    mask = df['qyteti'] == kodi
    n = (df.loc[mask, 'anomaly_global'] == -1).sum()
    t = mask.sum()
    print(f"   {emri:12s}: {n:5d} ({n/t*100:.1f}%)")

# MODEL B: Per-City (contamination e adaptuar per cdo qytet)
print("MODEL B: Per-City Isolation Forest (adaptive)")

# Adaptive contamination: based on IQR analysis per city
CITY_CONTAMINATION = {
    1: 0.07,   # Prishtine - more polluted, higher threshold
    2: 0.05,   # Prizren - moderate
    3: 0.03,   # Peje - cleanest, lower threshold
}

df['score_percity'] = 0.0
df['anomaly_percity'] = 1

for kodi, emri in qytetet_emrat.items():
    mask = df['qyteti'] == kodi
    df_city = df.loc[mask, FEATURES]
    contam = CITY_CONTAMINATION[kodi]

    scaler_city = StandardScaler()
    X_city = scaler_city.fit_transform(df_city)

    iso_city = IsolationForest(
        n_estimators=200, contamination=contam,
        max_samples='auto', random_state=42, n_jobs=-1
    )
    iso_city.fit(X_city)

    df.loc[mask, 'score_percity'] = iso_city.decision_function(X_city)
    df.loc[mask, 'anomaly_percity'] = iso_city.predict(X_city)

    n_anom = (df.loc[mask, 'anomaly_percity'] == -1).sum()
    total = mask.sum()
    print(f"   {emri:12s}: contam={contam:.0%}, anomali={n_anom:5d} ({n_anom/total*100:.1f}%)")

n_anom_percity = (df['anomaly_percity'] == -1).sum()
print(f"\n   Total anomali per-city: {n_anom_percity} ({n_anom_percity/len(df)*100:.1f}%)")

# SEVERITY SCORING
print("SEVERITY SCORING (Niveli i rrezikshmërisë)")

# Use per-city model scores for severity
anomaly_mask = df['anomaly_percity'] == -1
anomaly_scores = df.loc[anomaly_mask, 'score_percity']

# Percentile-based severity thresholds
p33 = anomaly_scores.quantile(0.33)
p66 = anomaly_scores.quantile(0.66)

df['severity'] = 'Normal'
df.loc[anomaly_mask & (df['score_percity'] >= p33), 'severity'] = 'Low'
df.loc[anomaly_mask & (df['score_percity'] < p33) & (df['score_percity'] >= p66), 'severity'] = 'Medium'
df.loc[anomaly_mask & (df['score_percity'] < p66), 'severity'] = 'High'

print("   Shpërndarja e severity:")
for sev in ['Normal', 'Low', 'Medium', 'High']:
    n = (df['severity'] == sev).sum()
    print(f"   {sev:8s}: {n:6d} ({n/len(df)*100:.1f}%)")

print("\n   Mesataret sipas severity:")
severity_stats = df.groupby('severity')[['pm10', 'pm2_5', 'temperature_2m', 'wind_speed_10m']].mean().round(2)
for idx in ['Normal', 'Low', 'Medium', 'High']:
    if idx in severity_stats.index:
        row = severity_stats.loc[idx]
        print(f"   {idx:8s}: PM10={row['pm10']:6.1f}, PM2.5={row['pm2_5']:6.1f}, "
              f"Temp={row['temperature_2m']:5.1f}°C, Wind={row['wind_speed_10m']:4.1f} m/s")

# KRAHASIMI: Global vs Per-City
print("KRAHASIMI: Global vs Per-City")

# Unique anomalies found only by one method
only_global = ((df['anomaly_global'] == -1) & (df['anomaly_percity'] == 1)).sum()
only_percity = ((df['anomaly_percity'] == -1) & (df['anomaly_global'] == 1)).sum()
both = ((df['anomaly_global'] == -1) & (df['anomaly_percity'] == -1)).sum()

print(f"   Vetem Global:    {only_global:5d}")
print(f"   Vetem Per-City:  {only_percity:5d}")
print(f"   Te dyja:         {both:5d}")
print(f"   Total unike:     {only_global + only_percity + both:5d}")

# Per city breakdown
print("\n   Krahasimi sipas qytetit:")
for kodi, emri in qytetet_emrat.items():
    mask = df['qyteti'] == kodi
    ng = (df.loc[mask, 'anomaly_global'] == -1).sum()
    np_ = (df.loc[mask, 'anomaly_percity'] == -1).sum()
    print(f"   {emri:12s}: Global={ng:5d}, Per-City={np_:5d}, Dif={np_-ng:+5d}")

# GRAFIKUT
OUTPUT_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Grafiku 1: PM2.5 vs PM10 me severity ---
fig, ax = plt.subplots(figsize=(12, 7))
normal_mask = df['severity'] == 'Normal'
for sev, color, sz, alpha in [('Normal', 'steelblue', 5, 0.1),
                                ('Low', '#FFA500', 12, 0.5),
                                ('Medium', '#FF4500', 18, 0.6),
                                ('High', '#DC143C', 25, 0.8)]:
    m = df['severity'] == sev
    n = m.sum()
    ax.scatter(df.loc[m, 'pm10'], df.loc[m, 'pm2_5'],
               c=color, alpha=alpha, s=sz, label=f'{sev} ({n})')
ax.set_xlabel('PM10 (µg/m³)', fontsize=12)
ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
ax.set_title('Anomali me Severity Scoring (Per-City Model)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'if_anomalies_severity.png'), dpi=150)
plt.close()

# --- Grafiku 2: Per-city anomaly comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, (kodi, emri) in enumerate(qytetet_emrat.items()):
    ax = axes[idx]
    mask = df['qyteti'] == kodi
    city_df = df[mask]
    nm = city_df['anomaly_percity'] == 1
    am = city_df['anomaly_percity'] == -1
    ax.scatter(city_df.loc[nm.values, 'pm10'], city_df.loc[nm.values, 'pm2_5'],
               c='steelblue', alpha=0.15, s=5, label=f'Normal ({nm.sum()})')
    ax.scatter(city_df.loc[am.values, 'pm10'], city_df.loc[am.values, 'pm2_5'],
               c='red', alpha=0.5, s=12, label=f'Anomali ({am.sum()})')
    contam = CITY_CONTAMINATION[kodi]
    ax.set_title(f'{emri} (contam={contam:.0%})', fontsize=13, fontweight='bold')
    ax.set_xlabel('PM10', fontsize=11)
    ax.set_ylabel('PM2.5', fontsize=11)
    ax.legend(fontsize=9)
plt.suptitle('Per-City Isolation Forest', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'if_anomalies_per_city.png'), dpi=150)
plt.close()

# --- Grafiku 3: Anomalite sipas muajit (Global vs Per-City) ---
anomalies_global = df[df['anomaly_global'] == -1]
anomalies_percity = df[df['anomaly_percity'] == -1]
monthly_global = anomalies_global.groupby('month').size()
monthly_percity = anomalies_percity.groupby('month').size()

fig, ax = plt.subplots(figsize=(12, 6))
month_names = ['Jan', 'Shk', 'Mar', 'Pri', 'Maj', 'Qer',
               'Kor', 'Gus', 'Sht', 'Tet', 'Nen', 'Dhj']
months = range(1, 13)
x = np.arange(12)
w = 0.35
counts_g = [monthly_global.get(m, 0) for m in months]
counts_p = [monthly_percity.get(m, 0) for m in months]
ax.bar(x - w/2, counts_g, w, label='Global (V1)', color='steelblue', alpha=0.8, edgecolor='black')
ax.bar(x + w/2, counts_p, w, label='Per-City (V2)', color='darkorange', alpha=0.8, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(month_names)
ax.set_xlabel('Muaji', fontsize=12)
ax.set_ylabel('Numri i Anomalive', fontsize=12)
ax.set_title('Krahasimi: Global vs Per-City sipas Muajit', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'if_anomalies_monthly_comparison.png'), dpi=150)
plt.close()

# --- Grafiku 4: Severity Heatmap per ora x muaji ---
fig, ax = plt.subplots(figsize=(14, 6))
high_medium = df[df['severity'].isin(['High', 'Medium'])]
if len(high_medium) > 0:
    pivot = high_medium.groupby(['month', 'hour']).size().unstack(fill_value=0)
    # Ensure all hours and months are present
    pivot = pivot.reindex(index=range(1, 13), columns=range(0, 24), fill_value=0)
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3)
    ax.set_xlabel('Ora e Dites', fontsize=12)
    ax.set_ylabel('Muaji', fontsize=12)
    ax.set_title('Heatmap: Anomali te Renda (Medium + High) sipas Ores dhe Muajit',
                 fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'if_severity_heatmap.png'), dpi=150)
plt.close()

# --- Grafiku 5: Anomaly Score Distribution per city ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (kodi, emri) in enumerate(qytetet_emrat.items()):
    ax = axes[idx]
    mask = df['qyteti'] == kodi
    nm = mask & (df['anomaly_percity'] == 1)
    am = mask & (df['anomaly_percity'] == -1)
    ax.hist(df.loc[nm, 'score_percity'], bins=50, alpha=0.7, color='steelblue',
            label='Normal', density=True)
    ax.hist(df.loc[am, 'score_percity'], bins=30, alpha=0.7, color='red',
            label='Anomali', density=True)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.set_title(f'{emri}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Anomaly Score', fontsize=10)
    ax.legend(fontsize=9)
plt.suptitle('Anomaly Score Distribution per Qytet', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'if_score_distribution_per_city.png'), dpi=150)
plt.close()

print(f"\nGrafikut u ruajten te: {OUTPUT_DIR}")

# EKSPORTI
export_cols = FEATURES + ['month', 'hour', 'day_of_week', 'qyteti',
                          'score_percity', 'anomaly_percity', 'severity']
export_path = os.path.join(BASE_DIR, 'Datasetet', 'ml_ready_dataset', 'anomalies_detected.csv')
anomalies_export = df[df['anomaly_percity'] == -1][export_cols]
anomalies_export.to_csv(export_path, index=False)
print(f"Anomalite u eksportuan te: {export_path}")
print(f"Total: {len(anomalies_export)} anomali")

# Ruajtja e rezultateve ne txt
MODEL_DIR = os.path.join(BASE_DIR, 'Modelet')
os.makedirs(MODEL_DIR, exist_ok=True)
results_path = os.path.join(MODEL_DIR, 'rezultatet_isolation_forest.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write('ISOLATION FOREST - REZULTATET\n')
    f.write(f'Dataseti: {len(df)} rreshta, {df.shape[1]} kolona\n\n')
    f.write('MODEL A (Global):\n')
    f.write(f'  Total anomali: {n_anom_global} ({n_anom_global/len(df)*100:.1f}%)\n')
    for kodi, emri in qytetet_emrat.items():
        mask = df['qyteti'] == kodi
        n = (df.loc[mask, 'anomaly_global'] == -1).sum()
        t = mask.sum()
        f.write(f'  {emri:12s}: {n:5d} ({n/t*100:.1f}%)\n')
    f.write('\nMODEL B (Per-City Adaptive):\n')
    f.write(f'  Total anomali: {n_anom_percity} ({n_anom_percity/len(df)*100:.1f}%)\n')
    for kodi, emri in qytetet_emrat.items():
        mask = df['qyteti'] == kodi
        n = (df.loc[mask, 'anomaly_percity'] == -1).sum()
        t = mask.sum()
        contam = CITY_CONTAMINATION[kodi]
        f.write(f'  {emri:12s}: contam={contam:.0%}, anomali={n:5d} ({n/t*100:.1f}%)\n')
    f.write('\nSeverity Scoring:\n')
    for sev in ['Normal', 'Low', 'High']:
        n = (df['severity'] == sev).sum()
        f.write(f'  {sev:8s}: {n:6d} ({n/len(df)*100:.1f}%)\n')
    if len(severity_stats) > 0:
        f.write('\nMesataret sipas severity:\n')
        for idx in ['Normal', 'Low', 'High']:
            if idx in severity_stats.index:
                row = severity_stats.loc[idx]
                f.write(f'  {idx:8s}: PM10={row["pm10"]:6.1f}, PM2.5={row["pm2_5"]:6.1f}, '
                        f'Temp={row["temperature_2m"]:5.1f}C, Wind={row["wind_speed_10m"]:4.1f} m/s\n')
print(f"Rezultatet u ruajten te: {results_path}")

# Eksporti JSON (per model_evaluation.py - krahasimi me modelet tjera)
# Per-city stats for JSON
city_stats = {}
for kodi, emri in qytetet_emrat.items():
    mask = df['qyteti'] == kodi
    total = mask.sum()
    n_global = (df.loc[mask, 'anomaly_global'] == -1).sum()
    n_percity = (df.loc[mask, 'anomaly_percity'] == -1).sum()
    city_stats[emri] = {
        'total_rows': int(total),
        'contamination': CITY_CONTAMINATION[kodi],
        'anomalies_global': int(n_global),
        'anomalies_percity': int(n_percity),
        'pct_global': round(n_global/total*100, 1),
        'pct_percity': round(n_percity/total*100, 1),
    }

severity_dict = {}
for sev in ['Normal', 'Low', 'High']:
    mask_sev = df['severity'] == sev
    n_sev = int(mask_sev.sum())
    if n_sev > 0 and sev in severity_stats.index:
        row = severity_stats.loc[sev]
        severity_dict[sev] = {
            'count': n_sev,
            'pct': round(n_sev/len(df)*100, 1),
            'mean_pm25': round(float(row['pm2_5']), 1),
            'mean_pm10': round(float(row['pm10']), 1),
        }

if_eval = {
    'model_name': 'Isolation Forest',
    'model_type': 'unsupervised',
    'author': 'Endrita',
    'task': 'anomaly_detection',
    'features_used': FEATURES,
    'dataset_rows': len(df),
    'total_anomalies_global': int(n_anom_global),
    'total_anomalies_percity': int(n_anom_percity),
    'pct_anomalies': round(n_anom_percity/len(df)*100, 1),
    'city_stats': city_stats,
    'severity': severity_dict,
    'overlap': {
        'only_global': int(only_global),
        'only_percity': int(only_percity),
        'both': int(both),
    }
}

json_path = os.path.join(MODEL_DIR, 'isolation_forest.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(if_eval, f, indent=2, ensure_ascii=False)
print(f"JSON per evaluim: {json_path}")

print("PERFUNDOI ME SUKSES!")
