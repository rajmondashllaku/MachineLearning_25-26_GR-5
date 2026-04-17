import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json  # SHTUAR PËR JSON


def trajner_kmeans(file_path):
    print("\n" + "=" * 50)
    print("TRAJNIMI I MODELIT: K-MEANS CLUSTERING")
    print("=" * 50)

    # 1. Leximi i të dhënave
    if not os.path.exists(file_path):
        print(f"[!] Gabim: Skedari nuk u gjet te {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f" -> Të dhënat u lexuan me sukses. Dimensioni: {df.shape}")

    # 2. Zgjedhja e veçorive për grupim
    kolonat_k_means = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'pm2_5']
    kolonat_ekzistuese = [col for col in kolonat_k_means if col in df.columns]
    X = df[kolonat_ekzistuese].copy()

    # 3. Shkallëzimi i të dhënave
    print(" -> Duke shkallëzuar të dhënat (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Krijimi i folderit për imazhe
    folderi_imazheve = '../../images'
    os.makedirs(folderi_imazheve, exist_ok=True)

    # 4. Metoda e Bërrylit
    print("\nDuke llogaritur Metodën e Bërrylit (1 deri në 10 grupe)...")
    wcss = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans_test = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans_test.fit(X_scaled)
        wcss.append(kmeans_test.inertia_)

    # Vizualizimi i Metodës së Bërrylit
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, marker='o', linestyle='--', color='#2c7bb6', linewidth=2, markersize=8)
    plt.title('Metoda e Bërrylit për Gjetjen e Grupeve Optimale', fontsize=14, fontweight='bold')
    plt.xlabel('Numri i Grupeve (K)', fontsize=12)
    plt.ylabel('WCSS (Distanca e gabimit)', fontsize=12)
    plt.xticks(K_range)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    shtegu_elbow = os.path.join(folderi_imazheve, 'kmeans_elbow_method.png')
    plt.savefig(shtegu_elbow, dpi=300)
    plt.close()
    print(f" -> Grafiku i Bërrylit u ruajt në: {shtegu_elbow}")

    # 5. Trajnimi Përfundimtar
    k_optimal = 4
    print(f"\nDuke trajnuar K-Means përfundimtar me {k_optimal} grupe...")
    kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 6. Analiza e Grupeve
    print("\n📊 PROFILET E ZBULUARA TË NDOTJES (Mesataret për grup):")
    profili = df.groupby('Cluster')[kolonat_ekzistuese].mean().round(2)
    print(profili)

    # 8. Vizualizimet
    print("\nKrijimi i grafikut 3D...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        df['temperature_2m'],
        df['wind_speed_10m'],
        df['pm2_5'],
        c=df['Cluster'],
        cmap='viridis',
        alpha=0.6,
        s=30
    )
    ax.set_title('Profilet e Ndotjes në 3D (Temp vs Erë vs PM2.5)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Temperatura (°C)', fontsize=10)
    ax.set_ylabel('Shpejtësia e Erës (m/s)', fontsize=10)
    ax.set_zlabel('Ndotja PM2.5 (µg/m³)', fontsize=10)
    legend = ax.legend(*scatter.legend_elements(), title='Grupi (Cluster)')
    ax.add_artist(legend)
    shtegu_3d = os.path.join(folderi_imazheve, 'kmeans_clusters_3d.png')
    plt.savefig(shtegu_3d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Grafiku 3D u ruajt në: {shtegu_3d}")

    # --- KËTU FILLON PJESA E SHTUAR PËR TË RUAJTUR NË JSON ---

    print("\nDuke ruajtur rezultatet e K-Means në JSON...")

    # Sigurohemi që folderi Modelet ekziston
    folderi_modeleve = '../../Modelet'
    os.makedirs(folderi_modeleve, exist_ok=True)

    # Konvertojmë profilin (DataFrame) në një fjalor për ta ruajtur në JSON
    # Kthimi i indekseve në string është i rëndësishëm që JSON ta kuptojë
    profili_dict = profili.to_dict(orient='index')
    profili_json_safe = {str(k): v for k, v in profili_dict.items()}

    rezultatet_kmeans = {
        "model_type": "unsupervised",
        "model_name": "K-Means Clustering",
        "n_clusters": k_optimal,
        "features_used": kolonat_ekzistuese,
        "cluster_profiles": profili_json_safe
    }

    shtegu_json = os.path.join(folderi_modeleve, 'kmeans_clustering.json')
    with open(shtegu_json, 'w', encoding='utf-8') as f:
        json.dump(rezultatet_kmeans, f, indent=4)

    print(f" -> Rezultatet u ruajtën në: {shtegu_json}")

    # (Pjesën tjetër të vizualizimeve Barplot, Boxplot, Pairplot i lashë të paprekura, vazhdojnë këtu poshtë)
    print("\nKrijimi i vizualizimit të tabelës së profileve...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Karakteristikat e Profileve të Ndotjes (Mesataret e Grupeve)', fontsize=16, fontweight='bold', y=1.02)
    profili['temperature_2m'].plot(kind='bar', ax=axes[0, 0], color='#ff9999', edgecolor='black')
    axes[0, 0].set_title('Temperatura (°C)', fontsize=12)
    axes[0, 0].set_ylabel('°C')
    profili['relative_humidity_2m'].plot(kind='bar', ax=axes[0, 1], color='#66b3ff', edgecolor='black')
    axes[0, 1].set_title('Lagështia Relative (%)', fontsize=12)
    axes[0, 1].set_ylabel('%')
    profili['wind_speed_10m'].plot(kind='bar', ax=axes[1, 0], color='#99ff99', edgecolor='black')
    axes[1, 0].set_title('Shpejtësia e Erës (m/s)', fontsize=12)
    axes[1, 0].set_ylabel('m/s')
    profili['pm2_5'].plot(kind='bar', ax=axes[1, 1], color='#ffcc99', edgecolor='black')
    axes[1, 1].set_title('Ndotja PM2.5 (µg/m³)', fontsize=12)
    axes[1, 1].set_ylabel('µg/m³')
    for ax in axes.flat:
        ax.set_xlabel('Grupi (Cluster)')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    shtegu_bar = os.path.join(folderi_imazheve, 'kmeans_cluster_profiles_bars.png')
    plt.savefig(shtegu_bar, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nKrijimi i Boxplots për shpërndarjen e plotë të të dhënave...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Shpërndarja dhe Ekstremet e Variablave sipas Grupeve', fontsize=16, fontweight='bold', y=1.02)
    sns.boxplot(x='Cluster', y='temperature_2m', data=df, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Temperatura (°C)')
    sns.boxplot(x='Cluster', y='relative_humidity_2m', data=df, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('Lagështia Relative (%)')
    sns.boxplot(x='Cluster', y='wind_speed_10m', data=df, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Shpejtësia e Erës (m/s)')
    sns.boxplot(x='Cluster', y='pm2_5', data=df, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Ndotja PM2.5 (µg/m³)')
    plt.tight_layout()
    shtegu_boxplot = os.path.join(folderi_imazheve, 'kmeans_cluster_boxplots.png')
    plt.savefig(shtegu_boxplot, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nKrijimi i Pairplot (Lidhjet e të gjitha variablave)...")
    df_plot = df[kolonat_ekzistuese + ['Cluster']].copy()
    df_plot['Cluster'] = df_plot['Cluster'].astype(str)
    pair_plot = sns.pairplot(df_plot, hue='Cluster', palette='viridis', corner=True, plot_kws={'alpha': 0.5})
    pair_plot.fig.suptitle('Matrica e Lidhjeve mes Motit dhe Ndotjes', y=1.02, fontsize=16, fontweight='bold')
    shtegu_pairplot = os.path.join(folderi_imazheve, 'kmeans_cluster_pairplot.png')
    pair_plot.savefig(shtegu_pairplot, dpi=300)


if __name__ == "__main__":
    skedari = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'
    trajner_kmeans(skedari)