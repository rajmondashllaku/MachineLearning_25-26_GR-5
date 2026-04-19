import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
import json

def trajner_xgboost(file_path):
    print("==================================================")
    print("TRAJNIMI I MODELIT GLOBAL: XGBOOST REGRESSOR")
    print("==================================================")

    if not os.path.exists(file_path):
        print(f"[!] Gabim: Skedari nuk u gjet te {file_path}")
        return

    # 1. Leximi i të dhënave globale
    df = pd.read_csv(file_path)
    print(f" -> Dimensioni origjinal i datasetit: {df.shape}")

    # (Është hequr One-Hot Encoding për të detyruar modelin të mësojë vetëm nga moti)

    # 2. Ndarja e veçorive (Features) dhe Target-it
    # SHTUAR: 'qyteti' në listën e fshirjes
    kolonat_per_fshirje = ['pm2_5', 'pm10', 'time', 'sezoni_i_ngrohjes', 'qyteti']
    X = df.drop(columns=[col for col in kolonat_per_fshirje if col in df.columns])
    y = df['pm2_5']

    kolonat_finale = list(X.columns)  # Ruajmë emrat e kolonave për JSON
    print(f" -> Numri i veçorive finale për trajnim: {len(kolonat_finale)}")

    # 3. Ndarja në Trajnim dhe Testim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f" -> Rreshta për Trajnim: {X_train.shape[0]}")
    print(f" -> Rreshta për Testim:  {X_test.shape[0]}\n")

    # 4. Shkallëzimi i të dhënave (StandardScaler)
    print(" -> Duke shkallëzuar të dhënat...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Rikthimi në DataFrame për të ruajtur emrat e kolonave për SHAP dhe grafikët
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=kolonat_finale)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=kolonat_finale)

    # 5. Trajnimi i Modelit XGBoost
    print("Duke trajnuar modelin XGBoost...")
    xg_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric=['rmse']
    )

    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    xg_model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)

    # 6. Parashikimi dhe Evaluimi
    y_pred = xg_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nREZULTATET E MODELIT GLOBAL:")
    print(f" * MAE (Gabimi Mesatar Absolut): {mae:.2f} µg/m³")
    print(f" * RMSE (Gabimi Mesatar Katror): {rmse:.2f} µg/m³")
    print(f" * R² (Saktësia):                {r2:.4f}")

    # ==========================================
    # VIZUALIZIMET
    # ==========================================
    folderi_imazheve = '../../images'
    os.makedirs(folderi_imazheve, exist_ok=True)
    print("\nDuke gjeneruar vizualizimet analitike...")

    # Vizualizimi 1: Feature Importance
    feature_importances = xg_model.feature_importances_
    fi_df = pd.DataFrame({
        'Veçoria': kolonat_finale,
        'Rëndësia': feature_importances
    }).sort_values(by='Rëndësia', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Rëndësia', y='Veçoria', data=fi_df, palette='viridis', hue='Veçoria', legend=False)
    plt.title('Rëndësia e Faktorëve në Parashikimin Global të PM2.5 (XGBoost)', fontsize=14, fontweight='bold')
    plt.xlabel('Pesha (Importance Score)')
    plt.ylabel('Veçoria (Feature)')
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_feature_importance2.png'), dpi=300)
    plt.close()

    # Vizualizimi 2: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='#2c7bb6', label='Parashikimet')
    max_val = max(max(y_test), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='#d7191c', linestyle='--', linewidth=2, label='Vija e Përsosmërisë')
    plt.title('Saktësia e Parashikimeve: Vlerat Reale vs Modeli Global', fontsize=14, fontweight='bold')
    plt.xlabel('Vlerat Reale të PM2.5 (µg/m³)')
    plt.ylabel('Parashikimet e Modelit (µg/m³)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_actual_vs_predicted2.png'), dpi=300)
    plt.close()

    # Vizualizimi 3: Learning Curve
    rezultatet_eval = xg_model.evals_result()
    numri_pemeve = len(rezultatet_eval['validation_0']['rmse'])
    x_axis = range(0, numri_pemeve)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, rezultatet_eval['validation_0']['rmse'], label='Gabimi në Trajnim', linewidth=2)
    plt.plot(x_axis, rezultatet_eval['validation_1']['rmse'], label='Gabimi në Testim', linewidth=2)
    plt.title('Kurba e të Mësuarit (XGBoost Global)', fontsize=14, fontweight='bold')
    plt.xlabel('Numri i Pemëve të Vendimit')
    plt.ylabel('Gabimi RMSE (µg/m³)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_learning_curve2.png'), dpi=300)
    plt.close()

    # Vizualizimi 4: Residuals (Shpërndarja e Gabimeve)
    mbetjet = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(mbetjet, kde=True, color='#9b59b6', bins=40)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Gabimi Zero')
    plt.title('Shpërndarja e Gabimeve të Parashikimit (Residuals)', fontsize=14, fontweight='bold')
    plt.xlabel('Diferenca (Reale - Parashikuar) µg/m³')
    plt.ylabel('Frekuenca (Numri i Ditëve)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_residuals_histogram2.png'), dpi=300)
    plt.close()

    # Vizualizimi 5: SHAP Summary
    print(" -> Duke gjeneruar analizën SHAP...")
    explainer = shap.TreeExplainer(xg_model)
    shap_values = explainer.shap_values(X_test_scaled)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, show=False)
    plt.title('Ndikimi i faktorëve globalë në Ndotjen PM2.5 (SHAP Values)', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # RUAJTJA E MODELIT DHE METRIKAVE NË JSON
    # ==========================================
    print("\nDuke ruajtur modelin global dhe të dhënat e shkallëzimit...")
    folderi_modeleve = '../../Modelet'
    os.makedirs(folderi_modeleve, exist_ok=True)

    # 1. Ruajtja e vetë modelit
    shtegu_modelit = os.path.join(folderi_modeleve, 'xgboost_global_model.json')
    xg_model.save_model(shtegu_modelit)

    # 2. Ruajtja e Scaler-it dhe Metrikave në JSON
    rezultatet_metrikat = {
        "model_type": "supervised",
        "model_name": "XGBoost Regressor",  # <--- Emri tani del vetëm si XGBoost Regressor
        "features_used": kolonat_finale,
        "scaler_values": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist()
        },
        "metrics": {
            "R2": float(r2),
            "MAE": float(mae),
            "RMSE": float(rmse)
        }
    }

    shtegu_metrikave = os.path.join(folderi_modeleve, 'xgboost_global_metrics.json')
    with open(shtegu_metrikave, 'w', encoding='utf-8') as f:
        json.dump(rezultatet_metrikat, f, indent=4)

    print(f" -> Modeli u ruajt në: {shtegu_modelit}")
    print(f" -> Metrikat dhe Scaler-i u ruajtën në: {shtegu_metrikave}")

    print("\n" + "=" * 50)
    print("PROCESI PERFUNDOI ME SUKSES!")
    print("=" * 50)

    return xg_model, y_test, y_pred


if __name__ == "__main__":
    skedari = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'
    trajner_xgboost(skedari)