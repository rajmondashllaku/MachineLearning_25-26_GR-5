import os
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# EVALUIMI PËRFUNDIMTAR I MODELEVE (MASTER THESIS)
# Lexon të gjitha rezultatet JSON dhe gjeneron krahasimet
# ==========================================

# Rrugët e direktorive
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / 'Modelet'
OUTPUT_DIR = BASE_DIR / 'images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("EVALUIMI KRAHASUES I MODELEVE (MACHINE LEARNING)")
print("=" * 50)


def load_json_results():
    """Lexon të gjithë skedarët JSON nga direktoria e Modeleve"""
    supervised_models = []
    unsupervised_models = []

    if not MODELS_DIR.exists():
        print(f"Gabim: Direktoria {MODELS_DIR} nuk ekziston.")
        return [], []

    for file_path in MODELS_DIR.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Kontroll i zgjeruar për llojin e modelit
                model_type = data.get('model_type', 'unknown').lower()

                # Kusht shumë më i gjerë për modelet Supervised (kap edhe XGBoost)
                is_supervised = (
                        model_type == 'supervised' or
                        'metrics' in data or
                        'r2_score' in data or
                        'R2' in data or
                        'MAE' in data or
                        'mae' in data
                )

                if is_supervised:
                    supervised_models.append(data)
                elif model_type == 'unsupervised' or 'unsupervised_method' in data or 'task' in data:
                    unsupervised_models.append(data)
        except Exception as e:
            print(f"Problem gjatë leximit të {file_path.name}: {e}")

    return supervised_models, unsupervised_models


def evaluate_supervised(models):
    """Gjeneron tabela dhe grafiqe krahasuese për modelet Supervised"""
    if not models:
        print("Nuk u gjetën modele Supervised për krahasim.")
        return

    print("\n1. KRAHASIMI I MODELEVE SUPERVISED (PARASHIKIMI I PM2.5)")
    print("-" * 60)

    # Krijimi i DataFrame për krahasim
    records = []
    for m in models:
        name = m.get('model_name', m.get('model', 'Unknown'))

        # Shikon nëse matjet janë brenda një fjalori 'metrics' ose direkt si çelësa bazë
        metrics = m.get('metrics', m)

        r2 = metrics.get('R2', metrics.get('r2_score', metrics.get('r2', 0)))
        mae = metrics.get('MAE', metrics.get('mae', 0))
        rmse = metrics.get('RMSE', metrics.get('rmse', 0))

        records.append({
            'Modeli': name,
            'R² Score': float(r2) if r2 is not None else 0.0,
            'MAE': float(mae) if mae is not None else 0.0,
            'RMSE': float(rmse) if rmse is not None else 0.0
        })

    df_metrics = pd.DataFrame(records).sort_values(by='R² Score', ascending=True)

    # Printimi në konsolë i tabelës
    print(df_metrics.to_string(index=False))

    # Ruajtja e tabelës si CSV
    df_metrics.to_csv(MODELS_DIR / 'supervised_comparison_table.csv', index=False)

    # ==========================================
    # GRAFIKU 1: Krahasimi i R² (Saktësia)
    # ==========================================
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(df_metrics))
    bars = plt.barh(df_metrics['Modeli'], df_metrics['R² Score'], color=colors)

    plt.xlabel('Saktësia (R² Score)', fontsize=12)
    plt.title('Krahasimi i Saktësisë së Modeleve Parashikuese (R²)', fontsize=14, fontweight='bold')

    # Rregullojmë limitin e X sipas vlerës maksimale për ta bërë të bukur
    max_r2 = df_metrics['R² Score'].max()
    plt.xlim(0, max(max_r2 + 0.15, 1.0))

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eval_r2_comparison.png', dpi=300)
    plt.close()

    # ==========================================
    # GRAFIKU 2: Krahasimi i Gabimeve (MAE & RMSE)
    # ==========================================
    df_errors = df_metrics.melt(id_vars=['Modeli'], value_vars=['MAE', 'RMSE'],
                                var_name='Metrika', value_name='Gabimi (µg/m³)')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Gabimi (µg/m³)', y='Modeli', hue='Metrika', data=df_errors, palette='rocket')
    plt.title('Krahasimi i Gabimeve të Parashikimit (Sa më pak, aq më mirë)', fontsize=14, fontweight='bold')
    plt.legend(title='Lloji i Gabimit')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eval_error_comparison.png', dpi=300)
    plt.close()

    print(f"   -> Grafiqet e Supervised u ruajtën në folderin 'images'")

def evaluate_unsupervised(models):
    """Përmbledh rezultatet për modelet Unsupervised"""
    if not models:
        print("Nuk u gjetën modele Unsupervised për krahasim.")
        return

    print("\n2. PËRMBLEDHJA E MODELEVE UNSUPERVISED (ZBULIMI)")
    print("-" * 60)

    with open(MODELS_DIR / 'unsupervised_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write("RAPORTI I MODELEVE UNSUPERVISED\n")
        f.write("=" * 40 + "\n\n")

        for m in models:
            name = m.get('model_name', m.get('unsupervised_method', 'Unknown Model'))
            print(f"👉 Analizimi: {name}")
            f.write(f"MODELI: {name}\n")
            f.write("-" * 20 + "\n")

            if 'PCA' in name.upper():
                comps = m.get('n_components', 'N/A')
                variance = m.get('cumulative_variance', [])
                if variance:
                    max_var = variance[-1]
                    print(f"   - Komponentët: {comps}")
                    print(f"   - Varianca e shpjeguar: {max_var:.2%}")
                    f.write(f"Komponentët e mbajtur: {comps}\n")
                    f.write(f"Varianca totale e shpjeguar: {max_var:.2%}\n")

            elif 'Isolation Forest' in name:
                anomalies = m.get('total_anomalies_percity', 'N/A')
                pct = m.get('pct_anomalies', 'N/A')
                print(f"   - Total anomali (Per-City): {anomalies} ({pct}%)")
                f.write(f"Anomali totale te gjetura: {anomalies} rreshta ({pct}% e datasetit)\n")

                # Detajet e severity nëse ekzistojnë
                severity = m.get('severity', {})
                if severity:
                    for sev, data in severity.items():
                        f.write(
                            f"   * {sev} Severity: {data['count']} raste (PM2.5 mesatar: {data['mean_pm25']} µg/m³)\n")

            elif 'K-Means' in name or 'KMeans' in name:
                clusters = m.get('n_clusters', 'N/A')
                print(f"   - Numri i Klasterave: {clusters}")
                f.write(f"Numri optimal i klasterave të zbuluar: {clusters}\n")

            f.write("\n")

    print(f"   -> Raporti Unsupervised u ruajt në: Modelet/unsupervised_summary_report.txt")


def main():
    # Ngarko të gjitha të dhënat JSON
    supervised_models, unsupervised_models = load_json_results()

    if not supervised_models and not unsupervised_models:
        print("KUJDES: Nuk u gjet asnjë skedar .json në folderin Modelet.")
        print("Ju lutem ekzekutoni fillimisht skriptat e trajnimit (p.sh. train_random_forest.py)")
        return

    # Proceso dhe vizualizo
    evaluate_supervised(supervised_models)
    evaluate_unsupervised(unsupervised_models)

    print("\n" + "=" * 50)
    print("EVALUIMI PËRFUNDOI SUKSESSHËM! 🎉")
    print("Shikoni direktorinë 'images' për grafiqet e reja krahasuese.")
    print("=" * 50)


if __name__ == '__main__':
    main()