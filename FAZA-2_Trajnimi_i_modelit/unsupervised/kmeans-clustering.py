import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import json
from sklearn.metrics import silhouette_score


def trajner_kmeans(file_path):
    print("\n" + "=" * 50)
    print("TRAJNIMI I MODELIT: K-MEANS CLUSTERING (ME RUAJTJE JSON)")
    print("=" * 50)

    # 1. Leximi i te dhenave
    if not os.path.exists(file_path):
        print(f"[!] Gabim: Skedari nuk u gjet te {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f" -> Te dhenat u lexuan me sukses. Dimensioni: {df.shape}")

    # 2. Zgjedhja e vecorive per grupim
    kolonat_k_means = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'pm2_5']
    kolonat_ekzistuese = [col for col in kolonat_k_means if col in df.columns]
    X = df[kolonat_ekzistuese].copy()

    # 3. Shkallezimi i te dhenave
    print(" -> Duke shkallezuar te dhenat (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Krijimi i folderit per imazhe
    folderi_imazheve = '../../images'
    os.makedirs(folderi_imazheve, exist_ok=True)

    # 4. Metoda e Berrylit
    print("\nDuke llogaritur Metoden e Berrylit (1 deri ne 10 grupe)...")
    wcss = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans_test = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans_test.fit(X_scaled)
        wcss.append(kmeans_test.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, wcss, marker='o', linestyle='--', color='#2c7bb6', linewidth=2, markersize=8)
    plt.title('Metoda e Berrylit per Gjetjen e Grupeve Optimale', fontsize=14, fontweight='bold')
    plt.xlabel('Numri i Grupeve (K)', fontsize=12)
    plt.ylabel('WCSS (Distanca e gabimit)', fontsize=12)
    plt.xticks(K_range)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    shtegu_elbow = os.path.join(folderi_imazheve, 'kmeans_elbow_method.png')
    plt.savefig(shtegu_elbow, dpi=300)
    plt.close()
    print(f" -> Grafiku i Berrylit u ruajt ne: {shtegu_elbow}")

    # 5. Silhouette Score (Moster 15k per efiçience)
    print("\nDuke llogaritur Silhouette Score per K=2 deri ne 6 (moster 15k rreshta)...")
    sil_scores = []
    K_range_sil = range(2, 7)

    np.random.seed(42)
    indices = np.random.choice(X_scaled.shape[0], 15000, replace=False)
    X_sample = X_scaled[indices]

    for k in K_range_sil:
        kmeans_sil = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels_sample = kmeans_sil.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels_sample)
        sil_scores.append(score)
        print(f" -> K={k} | Silhouette Score: {score:.4f}")

    # 6. Trajnimi Perfundimtar
    k_optimal = 4
    print(f"\nDuke trajnuar K-Means perfundimtar me {k_optimal} grupe...")
    kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 7. Analiza e Grupeve dhe Ruajtja JSON (Perfshire Scaler-in)
    print("\nPROFILET E ZBULUARA TE NDOTJES (Mesataret per grup):")
    profili = df.groupby('Cluster')[kolonat_ekzistuese].mean().round(2)
    print(profili)

    folderi_modeleve = '../../Modelet'
    os.makedirs(folderi_modeleve, exist_ok=True)

    profili_dict = profili.to_dict(orient='index')
    profili_json_safe = {str(k): v for k, v in profili_dict.items()}

    # SHTESA: Nxjerrja e parametrave te StandardScaler per JSON
    scaler_params_json = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "features": kolonat_ekzistuese
    }

    rezultatet_kmeans = {
        "model_type": "unsupervised",
        "model_name": "K-Means Clustering",
        "n_clusters": k_optimal,
        "features_used": kolonat_ekzistuese,
        "scaler_values": scaler_params_json,  # Ketu ruhet scaler-i
        "cluster_profiles": profili_json_safe
    }

    shtegu_json = os.path.join(folderi_modeleve, 'kmeans_clustering.json')
    with open(shtegu_json, 'w', encoding='utf-8') as f:
        json.dump(rezultatet_kmeans, f, indent=4)

    print(f" -> Modeli dhe Scaler-i u ruan ne format JSON te: {shtegu_json}")

    # 8. Vizualizimet
    print("\nKrijimi i grafikëve vizualë...")

    # 3D Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['temperature_2m'], df['wind_speed_10m'], df['pm2_5'],
                         c=df['Cluster'], cmap='viridis', alpha=0.6, s=30)
    ax.set_title('Profilet e Ndotjes ne 3D (Temp vs Ere vs PM2.5)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Temperatura (°C)')
    ax.set_ylabel('Shpejtesia e Eres (km/h)')
    ax.set_zlabel('Ndotja PM2.5 (µg/m³)')
    legend = ax.legend(*scatter.legend_elements(), title='Grupi (Cluster)')
    ax.add_artist(legend)
    plt.savefig(os.path.join(folderi_imazheve, 'kmeans_clusters_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Bar charts per profilet
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Karakteristikat e Profileve te Ndotjes (Mesataret e Grupeve)', fontsize=16, fontweight='bold', y=1.02)
    profili['temperature_2m'].plot(kind='bar', ax=axes[0, 0], color='#ff9999', edgecolor='black')
    profili['relative_humidity_2m'].plot(kind='bar', ax=axes[0, 1], color='#66b3ff', edgecolor='black')
    profili['wind_speed_10m'].plot(kind='bar', ax=axes[1, 0], color='#99ff99', edgecolor='black')
    profili['pm2_5'].plot(kind='bar', ax=axes[1, 1], color='#ffcc99', edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'kmeans_cluster_profiles_bars.png'), dpi=300)
    plt.close()

    print("\n" + "=" * 50)
    print("PROCESI PERFUNDOI ME SUKSES")
    print("=" * 50)


if __name__ == "__main__":
    # Sigurohuni qe shtegu i skedarit te jete i sakte per strukturen tuaj
    skedari = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'
    trajner_kmeans(skedari)