import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class SupervisedAirQualityModel:
    """
    Perdorimi Linear Regression dhe Ridge Regression si modele baze
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """Ngarkon te dhenat dhe ben pergatitjen baze"""
        print("=== NGARKIMI I TE DHENAVE ===")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Skedari nuk u gjet: {self.data_path}")

        self.data = pd.read_csv(self.data_path)
        print(f"Dataset-i u ngarkua me sukses: {self.data.shape[0]} rreshta, {self.data.shape[1]} kolona")
        print(f"Kolonat: {list(self.data.columns)}")

        return self

    def preprocess_data(self, target_column='pm2_5'):
        """Perpunon te dhenat per trajnim"""
        print("\n=== PERPUNIMI I TE DHENAVE ===")

        # Kontrollo nese kolona target ekziston
        if target_column not in self.data.columns:
            # Provo variante te ndryshme
            possible_targets = ['pm25', 'PM25', 'pm2.5', 'PM2.5']
            for col in possible_targets:
                if col in self.data.columns:
                    target_column = col
                    break
            else:
                raise ValueError(f"Kolona target '{target_column}' nuk u gjet. Kolonat e disponueshme: {list(self.data.columns)}")

        print(f"Kolona target: {target_column}")

        # Pasto te dhenat
        self.data = self.data.dropna()
        print(f"Pasi u hoqen vlerat null: {self.data.shape[0]} rreshta")

        # Shto is_weekend nese ka day_of_week
        if 'day_of_week' in self.data.columns:
            def is_weekend_value(value):
                if pd.isna(value):
                    return 0
                if isinstance(value, (int, np.integer, float, np.floating)):
                    day_num = int(value)
                    return 1 if day_num in {5, 6, 7, 0} else 0
                value_str = str(value).strip().lower()
                weekend_tokens = {
                    'sat', 'saturday', 'shtune', 'shtunë', 'e shtune',
                    'sun', 'sunday', 'diele', 'die', 'e diele'
                }
                if value_str in weekend_tokens:
                    return 1
                if value_str.isdigit() and int(value_str) in {5, 6, 7, 0}:
                    return 1
                return 0

            self.data['is_weekend'] = self.data['day_of_week'].apply(is_weekend_value)
            print("   -> Converted day_of_week into binary is_weekend feature (weekend=1, weekday=0)")
            self.data = self.data.drop(columns=['day_of_week'])

        # Perzgjidh kolonat numerike dhe kategorike
        numeric_features = []
        categorical_features = []

        for col in self.data.columns:
            if col == target_column:
                continue

         # FORCE month as categorical
            if col == 'month':
                categorical_features.append(col)
                continue

            if self.data[col].dtype in ['int64', 'float64']:
                if self.data[col].nunique() < 20 and col != 'is_weekend':
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)

        print(f"Vecori numerike: {numeric_features}")
        print(f"Vecori kategorike: {categorical_features}")

        if 'pm10' in numeric_features:
            numeric_features.remove('pm10')
            print("   -> Removed pm10 from numeric features")

        # Ndaj features dhe target
        feature_columns = numeric_features + categorical_features
        self.X = self.data[feature_columns]
        self.y = self.data[target_column]

        print(f"Dimensioni i features: {self.X.shape}")
        print(f"Dimensioni i target: {self.y.shape}")

        # Krijimi i preprocessor pipeline
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return self

    def split_data(self, test_size=0.2, random_state=42):
        """Ndaj te dhenat ne train dhe test"""
        print("\n=== NDAJA E TE DHENAVE ===")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        print(f"Train set: {self.X_train.shape[0]} rreshta")
        print(f"Test set: {self.X_test.shape[0]} rreshta")

        return self

    def train_models(self):
        """Trajnon modelet Linear Regression dhe Ridge Regression"""
        print("\n=== TRAJNIMI I MODELEVE ===")

        # Linear Regression
        print("Trajnim i Linear Regression...")
        lr_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', LinearRegression())
        ])

        lr_pipeline.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr_pipeline

        # Ridge Regression me Grid Search
        print("Trajnim i Ridge Regression...")
        ridge_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', Ridge())
        ])

        # Parametrat per Grid Search
        param_grid = {
            'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
        }

        ridge_grid = GridSearchCV(
            ridge_pipeline,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        ridge_grid.fit(self.X_train, self.y_train)
        self.models['Ridge Regression'] = ridge_grid.best_estimator_

        print(f"Ridge - Alpha optimale: {ridge_grid.best_params_['regressor__alpha']}")

        return self

    def evaluate_models(self):
        """Vlereson performancen e modeleve"""
        print("\n=== VLERESIMI I MODELEVE ===")

        metrics = {
            #'MSE': mean_squared_error,
            'MAE': mean_absolute_error,
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'R²': r2_score
        }

        for model_name, model in self.models.items():
            print(f"\n{model_name}:")

            # Parashikimet
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            # Metrikat per train
            print("  Train Set:")
            for metric_name, metric_func in metrics.items():
                score = metric_func(self.y_train, y_pred_train)
                print(f"    {metric_name}: {score:.4f}")

            # Metrikat per test
            print("  Test Set:")
            for metric_name, metric_func in metrics.items():
                score = metric_func(self.y_test, y_pred_test)
                print(f"    {metric_name}: {score:.4f}")

            # Cross-validation score
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=5, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            print(f"    CV RMSE: {cv_rmse:.4f} (+/- {np.sqrt(cv_scores.std()) * 2:.4f})")

            # Ruaj rezultatet
            self.results[model_name] = {
                'y_pred_test': y_pred_test,
                'metrics_test': {metric_name: metric_func(self.y_test, y_pred_test)
                               for metric_name, metric_func in metrics.items()},
                'cv_rmse': cv_rmse
            }

        return self

    def plot_results(self, output_dir='images'):
        """Vizualizon rezultatet"""
        print("\n=== VIZUALIZIMI I REZULTATEVE ===")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Scatter plot per parashikimet vs vlerat reale
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for i, (model_name, result) in enumerate(self.results.items()):
            ax = axes[i]

            # Scatter plot
            ax.scatter(self.y_test, result['y_pred_test'], alpha=0.6, color='blue', edgecolors='k')
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', linewidth=2, label='Linja ideale')

            ax.set_xlabel('Vlerat Reale PM2.5 (µg/m³)')
            ax.set_ylabel('Vlerat Parashikuara PM2.5 (µg/m³)')
            ax.set_title(f'{model_name}\nR² = {result["metrics_test"]["R²"]:.4f}')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ridge-linear-model_comparison_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> U ruajt: ridge-linear-model_comparison_scatter.png")

        # Bar plot per metrikat
        metrics_df = pd.DataFrame({
            model_name: result['metrics_test']
            for model_name, result in self.results.items()
        }).T

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df.plot(kind='bar', ax=ax, colormap='Set2')
        ax.set_title('Krahasimi i Metrikave te Modeleve')
        ax.set_ylabel('Vlera')
        ax.set_xlabel('Modeli')
        ax.legend(title='Metrika', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ridge-linear-model_metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> U ruajt: ridge-linear-model_metrics_comparison.png")

        return self

    def get_feature_importance(self):
        """Merr rendesine e vecorive per modelet lineare"""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE - Modelet Lineare")
        print("=" * 60)

        OUTPUT_DIR = "images"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for model_name, model in self.models.items():
            if hasattr(model.named_steps['regressor'], 'coef_'):

                # Merr emrat e vecorive pasi preprocessing
                feature_names = self._get_feature_names()

                # Merr koeficientet
                coefficients = model.named_steps['regressor'].coef_

                # Per modelet lineare: importance = |coef|
                importance_values = np.abs(coefficients)

                # DataFrame si Random Forest version
                feature_imp_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance_values
                })

                # Agragimi i dummies per feature origjinale
                categorical_columns = set(self.preprocessor.transformers_[1][2])
                def aggregate_feature(name):
                    if '_' in name:
                        prefix = name.split('_', 1)[0]
                        if prefix in categorical_columns:
                            return prefix
                    return name

                feature_imp_df['Feature'] = feature_imp_df['Feature'].apply(aggregate_feature)
                feature_imp_df = feature_imp_df.groupby('Feature', as_index=False)['Importance'].sum()
                feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)

               
                # PRINTIMI
              
                print(f"\n{model_name}:")
                for _, row in feature_imp_df.iterrows():
                    bar = "█" * int(row['Importance'] * 10)
                    print(f"   {row['Feature']:25s} {row['Importance']:.4f}  {bar}")

             
                # GRAFIKU 
               
                fig, ax = plt.subplots(figsize=(12, 7))

                colors = sns.color_palette("viridis", len(feature_imp_df.head(15)))

                bars = ax.barh(
                    feature_imp_df['Feature'].head(15),
                    feature_imp_df['Importance'].head(15),
                    color=colors
                )

                ax.set_xlabel('Importance', fontsize=12)
                ax.set_title(
                    f'Feature Importance - {model_name} (pa PM10)',
                    fontsize=14,
                    fontweight='bold'
                )

                ax.invert_yaxis()

                for bar, val in zip(bars, feature_imp_df['Importance'].head(15)):
                    ax.text(
                        bar.get_width() + 0.003,
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}',
                        va='center',
                        fontsize=9
                    )

                plt.tight_layout()

                save_name = os.path.join(
                    OUTPUT_DIR,
                    f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
                )
                plt.savefig(save_name, dpi=150)
                plt.close()

                print(f"   -> U ruajt: {save_name}")

    def _get_feature_names(self):
        """Merr emrat e vecorive pasi preprocessing"""
        try:
            # Merr emrat e veçorive nga preprocessor
            feature_names = []

            # Merr emrat nga numeric features
            num_features = self.preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names.extend(num_features)

            # Merr emrat nga categorical features
            cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(cat_features)

            return feature_names
        except:
            # Nese nuk mund te merren emrat, kthe indeksa
            return [f'feature_{i}' for i in range(len(self.models['Linear Regression'].named_steps['regressor'].coef_))]

    def save_models(self, output_dir='Modelet'):
        """Ruaj metrikat dhe rendesine e vecorive si JSON (pa pkl)."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for model_name, model in self.models.items():
            model_data = {
                'model_name': model_name,
                'model_type': model.named_steps['regressor'].__class__.__name__,
                'metrics': {},
                'feature_importance': [],
                'intercept': None,
                'alpha': None
            }

            if model_name in self.results:
                model_data['metrics'] = {
                    key: float(value)
                    for key, value in self.results[model_name]['metrics_test'].items()
                }
                model_data['cv_rmse'] = float(self.results[model_name].get('cv_rmse', np.nan))

            if hasattr(model.named_steps['regressor'], 'coef_'):
                feature_names = self._get_feature_names()
                coefficients = model.named_steps['regressor'].coef_
                model_data['intercept'] = float(model.named_steps['regressor'].intercept_)
                if hasattr(model.named_steps['regressor'], 'alpha'):
                    model_data['alpha'] = float(model.named_steps['regressor'].alpha)

                if len(feature_names) == len(coefficients):
                    model_data['feature_importance'] = [
                        {
                            'feature': feature,
                            'coefficient': float(coef),
                            'abs_coefficient': float(abs(coef))
                        }
                        for feature, coef in zip(feature_names, coefficients)
                    ]

            filename = f"{model_name.lower().replace(' ', '_')}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as json_file:
                json.dump(model_data, json_file, indent=2, ensure_ascii=False)
            print(f"Modeli u ruajt si JSON: {filepath}")

    def run_complete_pipeline(self):
        """Ekzekuton pipeline-in e plote"""
        try:
            self.load_data()
            self.preprocess_data()
            self.split_data()
            self.train_models()
            self.evaluate_models()
            self.plot_results()
            self.get_feature_importance()
            self.save_models()

            print("\n=== PIPELINE U KOMPLETUA ME SUKSES ===")
            print("Modelet e trajnuara dhe rezultatet u ruajten ne dosjet perkatese.")

        except Exception as e:
            print(f"Gabim gjate ekzekutimit: {str(e)}")
            raise


def main():
    """Funksioni kryesor"""
    # 1. Gjej në mënyrë dinamike rrënjën e projektit (3 nivele sipër këtij skripti)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 2. Ndrysho "Working Directory" që çdo shteg relativ të nisë nga rrënja e projektit
    os.chdir(BASE_DIR)

    # 3. Tani shtegu origjinal do të funksionojë në mënyrë perfekte!
    data_path = 'Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'

    # Krijo dhe ekzekuto modelin
    model = SupervisedAirQualityModel(data_path)
    model.run_complete_pipeline()


if __name__ == "__main__":
    main()