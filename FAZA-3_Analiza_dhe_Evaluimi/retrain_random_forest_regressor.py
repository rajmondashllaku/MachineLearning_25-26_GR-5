import json
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class AirQualityLinearPhase3:
    """
    Phase 3 - Linear & Ridge Regression
    """
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.preprocessor = None
        self.data_path = None

    def find_data_path(self):
        possible_paths = [
            'Datasetet/ml_ready_dataset/kosova_global_ml_data.csv',
            '../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv',
            '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv',
            '../../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                self.data_path = path
                print(f"Dataset found: {path}")
                return True
        try:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent.parent
            candidate = project_root / 'Datasetet' / 'ml_ready_dataset' / 'kosova_global_ml_data.csv'
            if candidate.exists():
                self.data_path = str(candidate)
                print(f"Dataset found: {self.data_path}")
                return True
        except:
            pass
        raise FileNotFoundError("Dataset not found!")

    def load_data(self):
        print("=== PHASE 3: LOADING DATA ===")
        self.find_data_path()
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.data.shape[0]:,} rows, {self.data.shape[1]} columns")
        return self

    def advanced_preprocessing(self, target_column='pm2_5'):
        print("\n=== ADVANCED PREPROCESSING & FEATURE ENGINEERING ===")
        self.data = self.data.dropna().copy()
        self.y_original = self.data[target_column]
        self.y = np.log1p(self.y_original)
        numeric_features = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure',
                           'wind_speed_10m', 'hour']
        categorical_features = ['month', 'sezoni_i_ngrohjes', 'qyteti']
        self.data['temp_humidity'] = self.data['temperature_2m'] * self.data['relative_humidity_2m']
        self.data['wind_temp'] = self.data['wind_speed_10m'] * self.data['temperature_2m']
        numeric_features += ['temp_humidity', 'wind_temp']
        poly_features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'hour']
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
        self.preprocessor = ColumnTransformer([
            ('num_poly', numeric_transformer, poly_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), [col for col in numeric_features if col not in poly_features])
        ])
        feature_cols = poly_features + categorical_features + [col for col in numeric_features if col not in poly_features]
        self.X = self.data[feature_cols]
        print(f"Final feature shape: {self.X.shape}")
        return self

    def split_data(self, test_size=0.2, random_state=42):
        print("\n=== SPLITTING DATA ===")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Train: {self.X_train.shape[0]:,} | Test: {self.X_test.shape[0]:,}")
        return self

    def train_models(self):
        print("\n=== RETRAINING MODELS ===")
        # Linear Regression
        print("Linear Regression...")
        lr_pipe = Pipeline([('preprocessor', self.preprocessor), ('regressor', LinearRegression())])
        lr_pipe.fit(self.X_train, self.y_train)
        self.models['Linear_Regression'] = lr_pipe

        # Ridge Regression
        ridge_pipe = Pipeline([('preprocessor', self.preprocessor), ('regressor', Ridge())])
        param_grid = {'regressor__alpha': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]}
        print("Tuning Ridge Regression...")
        grid = GridSearchCV(ridge_pipe, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        self.models['Ridge_Regression'] = grid.best_estimator_
        best_alpha = grid.best_params_['regressor__alpha']
        print(f"Best alpha for Ridge: {best_alpha}")
        if best_alpha < 0.01:
            print("WARNING: Alpha is very small -> Ridge is almost identical to Linear")
        return self

    def evaluate_models(self):
        print("\n=== MODEL EVALUATION (Original Scale) ===")
        for name, model in self.models.items():
            y_pred_log = model.predict(self.X_test)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(self.y_test)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            print(f"\n{name}:")
            print(f" MAE : {mae:.4f}")
            print(f" RMSE : {rmse:.4f}")
            print(f" R² : {r2:.4f}")
            self.results[name] = {'y_true': y_true, 'y_pred': y_pred, 'metrics': {'MAE': mae, 'RMSE': rmse, 'R²': r2}}
        return self

    def create_consistent_plots(self, output_dir='images'):
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for i, (name, res) in enumerate(self.results.items()):
            ax = axes[i]
            ax.scatter(res['y_true'], res['y_pred'], alpha=0.3, s=8, color='steelblue' if i==0 else 'darkorange')
            ax.plot([0, res['y_true'].max()], [0, res['y_true'].max()], 'r--', lw=2)
            ax.set_xlabel('Vlera Reale (PM2.5)')
            ax.set_ylabel('Vlera Parashikuar (PM2.5)')
            ax.set_title(f'{name}\nR² = {res["metrics"]["R²"]:.4f}', fontweight='bold')
        plt.suptitle('FAZA 3 Linear & Ridge: Actual vs Predicted', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'faza3_actual_vs_predicted_linear_ridge.png'), dpi=150)
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, (name, res) in enumerate(self.results.items()):
            residuals = res['y_true'] - res['y_pred']
            ax = axes[i]
            ax.hist(residuals, bins=80, color='steelblue' if i==0 else 'darkorange', edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Residuals (Actual - Predicted)')
            ax.set_ylabel('Frekuenca')
            ax.set_title(f'FAZA 3 Residuals - {name}\n(Mean={residuals.mean():.3f}, Std={residuals.std():.3f})',
                         fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'faza3_linear_ridge_residuals.png'), dpi=150)
        plt.close()

        metrics = ['MAE', 'RMSE', 'R²']
        vals_lr = [self.results['Linear_Regression']['metrics'][m] for m in metrics]
        vals_ridge = [self.results['Ridge_Regression']['metrics'][m] for m in metrics]
        fig, ax = plt.subplots(figsize=(9, 6))
        x = np.arange(len(metrics))
        w = 0.35
        ax.bar(x - w/2, vals_lr, w, label='Linear Regression', color='steelblue', alpha=0.9)
        ax.bar(x + w/2, vals_ridge, w, label='Ridge Regression', color='darkorange', alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title('FAZA 3 Krahasimi i Modeleve: Linear vs Ridge', fontweight='bold')
        ax.legend()
        for i, (v1, v2) in enumerate(zip(vals_lr, vals_ridge)):
            ax.text(i - w/2, v1 + 0.01, f'{v1:.3f}', ha='center')
            ax.text(i + w/2, v2 + 0.01, f'{v2:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'faza3_model_comparison_linear_ridge.png'), dpi=150)
        plt.close()
        return self

    def _get_feature_names(self):
        feature_names = []
        poly_input = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'hour']
        try:
            poly_names = self.preprocessor.named_transformers_['num_poly'].named_steps['poly'].get_feature_names_out(poly_input)
            feature_names.extend(poly_names)
        except: pass
        try:
            cat_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(cat_names)
        except: pass
        try:
            other = self.preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names.extend(other)
        except: pass
        return feature_names

    def get_feature_importance(self):
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE - Phase 3")
        print("="*70)
       
        OUTPUT_DIR = "images"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for model_name, model in self.models.items():
            feature_names = self._get_feature_names()
            imp = np.abs(model.named_steps['regressor'].coef_)
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': imp})
           
            def clean(name):
                if '_' in name and any(x in name for x in ['month','qyteti','sezoni']):
                    return name.split('_',1)[0]
                return name
           
            imp_df['Feature'] = imp_df['Feature'].apply(clean)
            imp_df = imp_df.groupby('Feature', as_index=False)['Importance'].sum()
            imp_df = imp_df.sort_values('Importance', ascending=False)
            
            # Print top features
            print(f"\n{model_name} (Top 12):")
            for _, row in imp_df.head(12).iterrows():
                bar = "█" * int(min(row['Importance'] * 15, 60))
                print(f" {row['Feature']:30s} {row['Importance']:8.4f} {bar}")

            # === Generate separate plot for each model ===
            fig, ax = plt.subplots(figsize=(12, 8))
            top15 = imp_df.head(15)
            colors = sns.color_palette("viridis", len(top15))
            
            bars = ax.barh(top15['Feature'], top15['Importance'], color=colors)
            
            ax.set_xlabel('Importance (Absolute Coefficient)', fontsize=12)
            ax.set_title(f'Feature Importance - {model_name.replace("_", " ")}\n(Phase 3)', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            # Add value labels
            for bar, val in zip(bars, top15['Importance']):
                ax.text(bar.get_width() + 0.003, 
                       bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', 
                       va='center', fontsize=9)

            plt.tight_layout()
            save_name = os.path.join(OUTPUT_DIR, f'faza3_feature_importance_{model_name.lower()}.png')
            plt.savefig(save_name, dpi=150, bbox_inches='tight')
            plt.close()
        
        

        return self

    def save_models(self, output_dir='Modelet'):
        os.makedirs(output_dir, exist_ok=True)
        for model_name, model in self.models.items():
            model_data = {
                'model_name': model_name,
                'phase': '3',
                'best_params': {'alpha': float(model.named_steps['regressor'].alpha)} if 'Ridge' in model_name else {},
                'test_metrics': self.results[model_name]['metrics']
            }
            filename = f"faza3_{model_name.lower().replace(' ', '_')}.json"
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)

    def run(self):
        try:
            self.load_data()
            self.advanced_preprocessing()
            self.split_data()
            self.train_models()
            self.evaluate_models()
            self.create_consistent_plots()
            self.get_feature_importance()
            self.save_models()
            print("\n" + "="*80)
            print("PHASE 3 COMPLETED SUCCESSFULLY!")
            print("="*80)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    phase3 = AirQualityLinearPhase3()
    phase3.run()
