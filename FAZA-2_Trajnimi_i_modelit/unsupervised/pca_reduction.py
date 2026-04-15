import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class UnsupervisedWeatherPCA:
  
    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = Path(__file__).resolve().parents[2] / 'Datasetet' / 'ml_ready_dataset' / 'kosova_global_ml_data.csv'
        else:
            self.data_path = Path(data_path)
        self.data = None
        self.features = None
        self.scaled_features = None
        self.pca = None
        self.pca_result = None
        self.pca_index = None

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Skedari nuk u gjet: {self.data_path}")

        self.data = pd.read_csv(self.data_path)
        print(f"Dataset-i u ngarkua me sukses: {self.data.shape[0]} rreshta, {self.data.shape[1]} kolona")
        print(f"Kolonat: {list(self.data.columns)}")
        return self

    def preprocess_data(self, include_weather_features=True):
        if self.data is None:
            raise RuntimeError("Duhet se pari te ngarkoni te dhenat me load_data()")

        weather_columns = [
            'temperature_2m',
            'relative_humidity_2m',
            'surface_pressure',
            'wind_speed_10m'
        ]

        if include_weather_features:
            self.features = [col for col in weather_columns if col in self.data.columns]
        else:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.features = [col for col in numeric_cols if col not in ['pm2_5', 'pm10']]

        if len(self.features) == 0:
            raise ValueError("Nuk u gjet asnje kolone te pershtatshme per PCA.")

        self.data = self.data.dropna(subset=self.features)
        self.pca_index = self.data.index
        self.scaled_features = StandardScaler().fit_transform(self.data[self.features])
        print(f"Kolonat per PCA: {self.features}")
        print(f"Dimensioni i te dhenave te perpunuara: {self.scaled_features.shape}")
        return self

    def run_pca(self, n_components=0.95):
        if self.scaled_features is None:
            raise RuntimeError("Duhet se pari te perpunoni te dhenat me preprocess_data()")

        self.pca = PCA(n_components=n_components)
        self.pca_result = self.pca.fit_transform(self.scaled_features)

        explained_ratio = self.pca.explained_variance_ratio_
        total_components = len(explained_ratio)
        print(f"PCA u ekzekutua me {total_components} komponenta.")
        print("Shperndarja e variances per komponent:")
        for i, ratio in enumerate(explained_ratio, start=1):
            print(f"  Komponenti {i}: {ratio:.4f}")

        cum_variance = np.cumsum(explained_ratio)
        for i, cum_ratio in enumerate(cum_variance, start=1):
            print(f"  Variance kumulative pas {i} komponentash: {cum_ratio:.4f}")

        return self

    def plot_explained_variance(self, output_dir='images'):
        if self.pca is None:
            raise RuntimeError("Duhet se pari te ekzekutoni PCA me run_pca()")

        os.makedirs(output_dir, exist_ok=True)
        explained_ratio = self.pca.explained_variance_ratio_
        cum_variance = np.cumsum(explained_ratio)

        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_ratio) + 1), explained_ratio, alpha=0.7, label='Variance per Komponent')
        plt.step(range(1, len(cum_variance) + 1), cum_variance, where='mid', color='red', label='Variance Kumulative')
        plt.xlabel('Komponenti Kryesor')
        plt.ylabel('Pjesa e Shpjeguar e Variances')
        plt.title('Shpjegimi i Variances ne PCA')
        plt.xticks(range(1, len(explained_ratio) + 1))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = Path(output_dir) / 'pca_explained_variance.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"   -> U ruajt: {filename}")
        return self

    def plot_pca_scatter(self, output_dir='images'):
        if self.pca_result is None:
            raise RuntimeError("Run PCA first.")

        os.makedirs(output_dir, exist_ok=True)

        df_plot = pd.DataFrame({
            'PC1': self.pca_result[:, 0],
            'PC2': self.pca_result[:, 1],
            'qyteti': self.data['qyteti'].values
        })

        plt.figure(figsize=(10, 7))
        for city in sorted(df_plot['qyteti'].unique()):
            city_data = df_plot[df_plot['qyteti'] == city]
            plt.scatter(city_data['PC1'], city_data['PC2'], alpha=0.5, label=f'Qyteti {city}')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Scatter Plot: PC1 vs PC2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = Path(output_dir) / 'pca_scatter_cities.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"   -> U ruajt: {filename}")

    def calculate_reconstruction_error(self):
        if self.pca is None:
            raise RuntimeError("Run PCA first.")

        reconstructed = self.pca.inverse_transform(self.pca_result)
        mse = np.mean((self.scaled_features - reconstructed) ** 2)

        print("\n=== PCA Reconstruction Error ===")
        print(f"Mean Reconstruction Error: {mse:.6f}")

        self.reconstruction_error = mse
        return mse

    def save_component_loadings(self, output_dir='Modelet'):
        if self.pca is None:
            raise RuntimeError("Duhet se pari te ekzekutoni PCA me run_pca()")

        os.makedirs(output_dir, exist_ok=True)
        loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.features,
            columns=[f'PC{i+1}' for i in range(self.pca.components_.shape[0])]
        )

        # filename = Path(output_dir) / 'pca_feature_loadings.csv'
        # loadings.to_csv(filename)
        # print(f"   -> U ruajt: {filename}")
        # return self

    def save_pca_model(self, output_dir='Modelet'):
        if self.pca is None:
            raise RuntimeError("Duhet se pari te ekzekutoni PCA me run_pca()")

        os.makedirs(output_dir, exist_ok=True)

        pca_state = {
            "n_components": int(self.pca.n_components_),
            "n_features_in_": int(getattr(self.pca, "n_features_in_", 0)),
            "components_": self.pca.components_.tolist(),
            "explained_variance_": self.pca.explained_variance_.tolist(),
            "explained_variance_ratio_": self.pca.explained_variance_ratio_.tolist(),
            "singular_values_": self.pca.singular_values_.tolist(),
            "mean_": self.pca.mean_.tolist() if hasattr(self.pca, "mean_") else None,
            "noise_variance_": getattr(self.pca, "noise_variance_", None),
            "feature_names_in_": getattr(self.pca, "feature_names_in_", None).tolist() if hasattr(self.pca, "feature_names_in_") else None,
        }

        filename = Path(output_dir) / 'pca_reduction.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(pca_state, f, indent=2)

        print(f"   -> U ruajt: {filename} (si JSON)")
        return self

    def save_pca_results_for_evaluation(self, output_dir='Modelet'):
        """Ruaj rezultatet PCA për model_evaluation.py"""
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "model_name": "PCA_Unsupervised",
            "unsupervised_method": "PCA",
            "n_components": int(self.pca.n_components_),
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(self.pca.explained_variance_ratio_).tolist(),
            "reconstruction_error": float(self.reconstruction_error) if hasattr(self, 'reconstruction_error') else None,
            "features_used": self.features,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        filename = Path(output_dir) / 'pca_results.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"   -> U ruajt për krahasim në model_evaluation.py: {filename}")
        return filename

    def run_complete_pipeline(self, n_components=0.95):
        self.load_data()
        self.preprocess_data()
        self.run_pca(n_components=n_components)
        self.plot_explained_variance()
        self.plot_pca_scatter()
        self.calculate_reconstruction_error()
        self.save_component_loadings()
        self.save_pca_model()
        self.save_pca_results_for_evaluation()

        print("\n=== PCA PERFUNDIMI ME SUKSES ===")
        return self


def main():
    model = UnsupervisedWeatherPCA()
    model.run_complete_pipeline()


if __name__ == '__main__':
    main()
