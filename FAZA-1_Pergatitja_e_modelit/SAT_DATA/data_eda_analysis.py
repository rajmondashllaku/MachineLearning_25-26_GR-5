import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def vizualizo_korrelacionin(df, qyteti, output_dir):
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f'Matrica e Korrelacionit - {qyteti.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'korrelacioni_{qyteti}.png'), dpi=300)
    plt.close()
    print(f"   -> U ruajt: korrelacioni_{qyteti}.png")


def vizualizo_ciklin_ditor(df, qyteti, output_dir):
    plt.figure(figsize=(12, 6))
    if 'hour' in df.columns and 'pm2_5' in df.columns:
        trendi_pm25 = df.groupby('hour')['pm2_5'].mean()
        plt.plot(trendi_pm25.index, trendi_pm25.values, marker='o', color='crimson', linewidth=2, label='PM 2.5')

        if 'pm10' in df.columns:
            trendi_pm10 = df.groupby('hour')['pm10'].mean()
            plt.plot(trendi_pm10.index, trendi_pm10.values, marker='s', color='steelblue', linewidth=2, linestyle='--',
                     label='PM 10')

        plt.title(f'Cikli Ditor i Ndotjes- {qyteti.upper()}', fontsize=14, fontweight='bold')
        plt.xlabel('Ora e Dites (00:00 - 23:00)', fontsize=12)
        plt.ylabel('Perqendrimi Mesatar (µg/m³)', fontsize=12)
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trendi_orar_{qyteti}.png'), dpi=300)
        plt.close()
        print(f"   -> U ruajt: trendi_orar_{qyteti}.png")
    else:
        print(f"   [!] Kujdes: Mungon kolona 'hour' ose 'pm2_5' për {qyteti}.")

def vizualizo_korrelacionin_global(file_path, output_dir):
    if not os.path.exists(file_path):
        print(f"\n   [!] Skedari global nuk u gjet te: {file_path}")
        return

    df = pd.read_csv(file_path)
    plt.figure(figsize=(12, 10))
    corr = df.corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 10})
    plt.title('Matrica e Korrelacionit - Kosove', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'korrelacioni_global.png'), dpi=300)
    plt.close()
    print(f"   -> U ruajt: korrelacioni_global.png")


def vizualizo_global(file_path, output_dir):
    if not os.path.exists(file_path):
        print(f"   [!] Skedari {file_path} nuk u gjet. Kontrollo shtegun!")
        return

    df = pd.read_csv(file_path)
    emrat_qyteteve = {1: 'Prishtine', 2: 'Prizren', 3: 'Pejë'}
    df['Emri_Qytetit'] = df['qyteti'].map(emrat_qyteteve)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Emri_Qytetit', y='pm2_5', data=df, hue='Emri_Qytetit', palette='Set2', legend=False)
    plt.title('Shperndarja e Ndotjes (PM2.5) sipas Qyteteve', fontsize=14, fontweight='bold')
    plt.xlabel('Qyteti', fontsize=12)
    plt.ylabel('Perqendrimi i PM2.5 (µg/m³)', fontsize=12)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'krahasimi_qyteteve_boxplot.png')
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"   -> U ruajt: {os.path.basename(boxplot_path)}")

    plt.figure(figsize=(12, 6))
    ngjyrat = {1: 'crimson', 2: 'steelblue', 3: 'forestgreen'}

    for kodi in [1, 2, 3]:
        df_qyteti = df[df['qyteti'] == kodi]
        if not df_qyteti.empty:
            trendi = df_qyteti.groupby('hour')['pm2_5'].mean()
            plt.plot(trendi.index, trendi.values, marker='o',
                     color=ngjyrat[kodi], linewidth=2.5, label=emrat_qyteteve[kodi])

    plt.title('Cikli Ditor i Ndotjes - Krahasim Rajonal (PM2.5)', fontsize=14, fontweight='bold')
    plt.xlabel('Ora e Dites (00:00 - 23:00)', fontsize=12)
    plt.ylabel('Perqendrimi Mesatar (µg/m³)', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Qytetet")
    plt.tight_layout()
    linechart_path = os.path.join(output_dir, 'trendi_orar_global.png')
    plt.savefig(linechart_path, dpi=300)
    plt.close()
    print(f"   -> U ruajt: {os.path.basename(linechart_path)}")


if __name__ == "__main__":
    qytetet = ['prishtine', 'prizren', 'peje']

    follderi_imazheve = '../../images'
    skedari_global = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'

    os.makedirs(follderi_imazheve, exist_ok=True)

    print("========================================")
    print("GJENERIMI I VIZUALIZIMEVE (EDA)")
    print("========================================")

    for qyteti in qytetet:
        skedari_hyrje = f'../../Datasetet/ml_ready_dataset/{qyteti}_ml_data.csv'

        if not os.path.exists(skedari_hyrje):
            print(f"\n[!] Skedari per {qyteti.upper()} nuk u gjet. Po e kalojme...")
            continue

        print(f"\nDuke procesuar grafiket per: {qyteti.upper()}")
        df = pd.read_csv(skedari_hyrje)

        vizualizo_korrelacionin(df, qyteti, follderi_imazheve)
        vizualizo_ciklin_ditor(df, qyteti, follderi_imazheve)

    print(f"\nDuke procesuar grafiket per: Kosove")
    vizualizo_korrelacionin_global(skedari_global, follderi_imazheve)
    vizualizo_global(skedari_global, follderi_imazheve)

    print("\n========================================")
    print("Te gjitha vizualizimet perfunduan me sukses! Shiko follderin 'images/'.")