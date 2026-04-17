import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


def trajner_xgboost_pa_outliers(file_path):
    print("==================================================")
    print("🚀 XGBOOST REGRESSOR (TEST: PA OUTLIERS > 150)")
    print("==================================================")

    if not os.path.exists(file_path):
        print(f"[!] Gabim: Skedari nuk u gjet te {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f" -> Dimensioni fillestar: {df.shape}")

    # ==========================================
    # TESTI I OUTLIERS (Fshirja e PM2.5 > 150)
    # ==========================================
    rreshtat_para = len(df)
    df = df[df['pm2_5'] <= 150]
    rreshtat_pas = len(df)

    if rreshtat_para != rreshtat_pas:
        print(f" ⚠️ U fshinë {rreshtat_para - rreshtat_pas} rreshta ekstreme (>150 µg/m³)")
    print(f" -> Dimensioni pas pastrimit: {df.shape}\n")
    # ==========================================

    kolonat_per_fshirje = ['pm2_5', 'pm10', 'time', 'sezoni_i_ngrohjes']
    X = df.drop(columns=[col for col in kolonat_per_fshirje if col in df.columns])
    y = df['pm2_5']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f" -> Rreshta per Trajnim: {X_train.shape[0]}")
    print(f" -> Rreshta per Testim:  {X_test.shape[0]}\n")

    print("Duke trajnuar modelin XGBoost (Pa Outliers)...")
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

    eval_set = [(X_train, y_train), (X_test, y_test)]
    xg_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    y_pred = xg_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 REZULTATET E MODELIT (PA OUTLIERS):")
    print(f" * MAE (Gabimi Mesatar Absolut): {mae:.2f} µg/m³")
    print(f" * RMSE (Gabimi Mesatar Katror): {rmse:.2f} µg/m³")
    print(f" * R² (Saktësia):                {r2:.4f}")

    folderi_imazheve = '../../images'
    os.makedirs(folderi_imazheve, exist_ok=True)
    print("\n📸 Duke gjeneruar vizualizimet analitike...")

    # Vizualizimi 1: Rëndësia e Veçorive
    feature_importances = xg_model.feature_importances_
    fi_df = pd.DataFrame({
        'Veçoria': X.columns,
        'Rëndësia': feature_importances
    }).sort_values(by='Rëndësia', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Rëndësia', y='Veçoria', data=fi_df, palette='viridis', hue='Veçoria', legend=False)
    plt.title('Rëndësia e Faktorëve (XGBoost - Pa Outliers)', fontsize=14, fontweight='bold')
    plt.xlabel('Pesha (Importance Score)', fontsize=12)
    plt.ylabel('Veçoria (Feature)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_feature_importance_no_outliers.png'), dpi=300)
    plt.close()

    # Vizualizimi 2: Vlerat Reale vs Parashikimet
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='#2c7bb6', label='Parashikimet')
    max_val = max(max(y_test), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='#d7191c', linestyle='--', linewidth=2, label='Vija e Përsosmërisë')
    plt.title('Saktësia e XGBoost (Pa Outliers): Reale vs Parashikuar', fontsize=14, fontweight='bold')
    plt.xlabel('Vlerat Reale të PM2.5 (µg/m³)', fontsize=12)
    plt.ylabel('Parashikimet e Modelit (µg/m³)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_actual_vs_predicted_no_outliers.png'), dpi=300)
    plt.close()

    # Vizualizimi 3: Kurba e të Mësuarit
    rezultatet_eval = xg_model.evals_result()
    numri_pemeve = len(rezultatet_eval['validation_0']['rmse'])
    x_axis = range(0, numri_pemeve)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, rezultatet_eval['validation_0']['rmse'], label='Gabimi në Trajnim', linewidth=2)
    plt.plot(x_axis, rezultatet_eval['validation_1']['rmse'], label='Gabimi në Testim', linewidth=2)
    plt.title('Kurba e të Mësuarit (XGBoost - Pa Outliers)', fontsize=14, fontweight='bold')
    plt.xlabel('Numri i Pemëve të Vendimit', fontsize=12)
    plt.ylabel('Gabimi RMSE (µg/m³)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_learning_curve_no_outliers.png'), dpi=300)
    plt.close()

    print(f" -> Të 3 grafikët u ruajtën me sukses në folderin: {folderi_imazheve}")

    # 7. Ruajtja e Modelit (JSON)
    print("\n💾 Duke ruajtur modelin...")
    folderi_modeleve = '../../modelet'
    os.makedirs(folderi_modeleve, exist_ok=True)

    shtegu_modelit = os.path.join(folderi_modeleve, 'xgboost_rajmonda_no_outliers.json')
    xg_model.save_model(shtegu_modelit)
    print(f" -> Modeli u ruajt me sukses në: {shtegu_modelit}")

    return xg_model, y_test, y_pred


if __name__ == "__main__":
    skedari = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'
    trajner_xgboost_pa_outliers(skedari)