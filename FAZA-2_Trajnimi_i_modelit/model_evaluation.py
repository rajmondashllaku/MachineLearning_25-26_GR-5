import os
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / 'Modelet'
OUTPUT_DIR = BASE_DIR / 'images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("EVALUIMI KRAHASUES I MODELEVE (MACHINE LEARNING)")
print("=" * 60)


def load_json_results():
    supervised_models = []
    unsupervised_models = []

    if not MODELS_DIR.exists():
        print(f"[!] Gabim: Direktoria {MODELS_DIR} nuk ekziston.")
        return [], []

    for file_path in MODELS_DIR.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                model_type = data.get('model_type', 'unknown').lower()

                is_supervised = (
                        model_type == 'supervised' or
                        'metrics' in data or
                        any(key in data for key in ['r2_score', 'R2', 'MAE', 'mae'])
                )

                if is_supervised:
                    supervised_models.append(data)
                elif model_type == 'unsupervised' or 'unsupervised_method' in data or 'task' in data:
                    unsupervised_models.append(data)
        except Exception as e:
            print(f"[!] Problem gjatë leximit të skedarit {file_path.name}: {e}")

    return supervised_models, unsupervised_models


def evaluate_supervised(models):
    if not models:
        print(" -> Nuk u gjetën modele Supervised për krahasim.")
        return

    print("\n1. KRAHASIMI I MODELEVE SUPERVISED (PARASHIKIMI I PM2.5)")
    print("-" * 60)

    records = []
    for m in models:
        name = m.get('model_name', m.get('model', 'Model i Panjohur'))
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

    print(df_metrics.to_string(index=False))
    df_metrics.to_csv(MODELS_DIR / 'supervised_comparison_table.csv', index=False)

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(df_metrics))
    bars = plt.barh(df_metrics['Modeli'], df_metrics['R² Score'], color=colors, edgecolor='black', alpha=0.9)

    plt.xlabel('Saktësia (R² Score) -> Më e lartë është më mirë', fontsize=12)
    plt.title('Krahasimi i Saktësisë së Modeleve Parashikuese (R²)', fontsize=14, fontweight='bold')

    max_r2 = df_metrics['R² Score'].max()
    plt.xlim(0, min(max_r2 + 0.15, 1.0))

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', va='center', fontsize=11, fontweight='bold', color='#333333')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eval_r2_comparison4.png', dpi=300)
    plt.close()

    df_errors = df_metrics.melt(id_vars=['Modeli'], value_vars=['MAE', 'RMSE'],
                                var_name='Metrika', value_name='Gabimi (µg/m³)')

    plt.figure(figsize=(12, 7))
    error_plot = sns.barplot(x='Gabimi (µg/m³)', y='Modeli', hue='Metrika', data=df_errors, palette='rocket',
                             edgecolor='black', alpha=0.9)

    plt.title('Krahasimi i Gabimeve të Parashikimit -> Më e ulët është më mirë', fontsize=14, fontweight='bold')
    plt.legend(title='Lloji i Gabimit', loc='lower right')

    for p in error_plot.patches:
        width = p.get_width()
        if width > 0:
            plt.text(width + 0.1, p.get_y() + p.get_height() / 2,
                     f'{width:.2f}', va='center', fontsize=10, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eval_error_comparison4.png', dpi=300)
    plt.close()

    print(f"\n   -> Grafiqet e Modeleve Supervised u ruajtën në '{OUTPUT_DIR.name}'")


def evaluate_unsupervised(models):
    if not models:
        print(" -> Nuk u gjetën modele Unsupervised për krahasim.")
        return

    print("\n2. PËRMBLEDHJA E MODELEVE UNSUPERVISED (ZBULIMI I STRUKTURAVE)")
    print("-" * 60)

    raporti_shtegu = MODELS_DIR / 'unsupervised_summary_report.txt'

    with open(raporti_shtegu, 'w', encoding='utf-8') as f:
        f.write("RAPORTI I MODELEVE UNSUPERVISED\n")
        f.write("=" * 50 + "\n\n")

        for m in models:
            name = m.get('model_name', m.get('unsupervised_method', 'Unknown Model'))
            print(f" Analizimi i parametrave nga: {name}")
            f.write(f"MODELI: {name}\n")
            f.write("-" * 30 + "\n")

            # Analiza për PCA
            if 'PCA' in name.upper():
                comps = m.get('n_components', 'N/A')
                variance = m.get('cumulative_variance', [])
                if variance:
                    max_var = variance[-1]
                    f.write(f" * Komponentët e mbajtur: {comps}\n")
                    f.write(f" * Varianca totale e shpjeguar: {max_var:.2%}\n")
                    f.write(
                        " * Shpjegimi: Modeli arriti të reduktojë dimensionet duke mbajtur shumicën e informacionit.\n")

            elif 'Isolation Forest' in name:
                anomalies = m.get('total_anomalies_percity', 'N/A')
                pct = m.get('pct_anomalies', 'N/A')
                f.write(f" * Anomali totale të gjetura: {anomalies} raste ({pct}% e datasetit)\n")

                severity = m.get('severity', {})
                if severity:
                    f.write(" * Nivelet e ashpërsisë së anomalive:\n")
                    for sev, data in severity.items():
                        f.write(
                            f"    - {sev}: {data.get('count', 0)} raste (Ndotja mesatare PM2.5: {data.get('mean_pm25', 0):.1f} µg/m³)\n")

            elif 'K-Means' in name or 'KMeans' in name:
                clusters = m.get('n_clusters', 'N/A')
                f.write(f" * Numri optimal i klasterave të zbuluar: {clusters}\n")

                profiles = m.get('cluster_profiles', {})
                if profiles:
                    f.write("\n * Karakteristikat e Profileve (Mesataret sipas grupit):\n")
                    for cluster_id, features in profiles.items():
                        f.write(f"    [Grupi {cluster_id}]: ")
                        features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
                        f.write(f"{features_str}\n")

            f.write("\n" + "=" * 50 + "\n\n")

    print(f"   -> Raporti Unsupervised u gjenerua me sukses te: Modelet/{raporti_shtegu.name}")


def main():
    supervised_models, unsupervised_models = load_json_results()

    if not supervised_models and not unsupervised_models:
        print("\n[!] KUJDES: Nuk u gjet asnjë skedar .json në folderin Modelet.")
        print("[!] Ju lutem ekzekutoni fillimisht skriptat e trajnimit në folderin FAZA-2.")
        return

    evaluate_supervised(supervised_models)
    evaluate_unsupervised(unsupervised_models)

    print("\n" + "=" * 60)
    print("EVALUIMI PËRFUNDOI SUKSESSHËM!")
    print("Hapi i radhës: Integrimi i modeleve në Inference Pipeline.")
    print("=" * 60)


if __name__ == '__main__':
    main()