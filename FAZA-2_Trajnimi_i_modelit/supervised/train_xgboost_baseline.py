import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import json


def trajner_xgboost(file_path):
    print("==================================================")
    print("TRAJNIMI I MODELIT: XGBOOST REGRESSOR")
    print("==================================================")

    # 1. Leximi i të dhënave
    if not os.path.exists(file_path):
        print(f"[!] Gabim: Skedari nuk u gjet te {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f" -> Të dhënat u lexuan me sukses. Dimensioni: {df.shape}")

    # 2. Përgatitja e veçorive (X) dhe targetit (y)
    kolonat_per_fshirje = ['pm2_5', 'pm10', 'time']

    X = df.drop(
        columns=[col for col in kolonat_per_fshirje if col in df.columns]
    )

    y = df['pm2_5']

    # 3. Ndarja në Trajnim dhe Testim
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f" -> Rreshta për Trajnim: {X_train.shape[0]}")
    print(f" -> Rreshta për Testim:  {X_test.shape[0]}\n")

    # 4. Ndërtimi i Modelit
    print("Duke trajnuar modelin XGBoost...")

    xg_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xg_model.fit(X_train, y_train)

    # 5. Parashikimi
    y_pred = xg_model.predict(X_test)

    # 6. Metrikat
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    print("\nREZULTATET E MODELIT:")
    print(f" * MAE  : {mae:.2f} µg/m³")
    print(f" * RMSE : {rmse:.2f} µg/m³")
    print(f" * R²   : {r2:.4f}")

    # ==========================================
    # KRIJIMI I FOLDERIT
    # ==========================================
    os.makedirs('rezultatet_rajmonda', exist_ok=True)

    # ==========================================
    # VIZUALIZIMI 1: Feature Importance
    # ==========================================
    print("\n -> Duke gjeneruar Feature Importance...")

    feature_importances = xg_model.feature_importances_

    fi_df = pd.DataFrame({
        'Veçoria': X.columns,
        'Rëndësia': feature_importances
    }).sort_values(by='Rëndësia', ascending=False)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x='Rëndësia',
        y='Veçoria',
        data=fi_df,
        palette='viridis',
        hue='Veçoria',
        legend=False
    )

    plt.title(
        'Rëndësia e Faktorëve në Parashikimin e PM2.5',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel('Pesha (Importance Score)', fontsize=12)
    plt.ylabel('Veçoria (Feature)', fontsize=12)

    plt.tight_layout()

    plt.savefig(
        'rezultatet_rajmonda/xgboost_feature_importance.png',
        dpi=300
    )

    plt.close()

    # ==========================================
    # VIZUALIZIMI 2: Actual vs Predicted
    # ==========================================
    print(" -> Duke gjeneruar Actual vs Predicted...")

    plt.figure(figsize=(8, 8))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.4,
        color='#2c7bb6',
        label='Parashikimet'
    )

    max_val = max(max(y_test), max(y_pred))

    plt.plot(
        [0, max_val],
        [0, max_val],
        color='#d7191c',
        linestyle='--',
        linewidth=2,
        label='Vija Ideale'
    )

    plt.title(
        'Vlerat Reale vs Parashikimet',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel(
        'Vlerat Reale të PM2.5 (µg/m³)',
        fontsize=12
    )

    plt.ylabel(
        'Parashikimet e Modelit (µg/m³)',
        fontsize=12
    )

    plt.legend()

    plt.grid(
        True,
        linestyle=':',
        alpha=0.6
    )

    plt.tight_layout()

    plt.savefig(
        'rezultatet_rajmonda/xgboost_actual_vs_predicted.png',
        dpi=300
    )

    plt.close()

    # ==========================================
    # VIZUALIZIMI 3: Residuals Histogram
    # ==========================================
    print(" -> Duke gjeneruar Residuals Histogram...")

    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))

    sns.histplot(
        residuals,
        kde=True,
        bins=40,
        color='#9b59b6'
    )

    plt.axvline(
        0,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Gabimi Zero'
    )

    plt.title(
        'Shpërndarja e Gabimeve të Modelit',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel(
        'Gabimi (Reale - Parashikuar)',
        fontsize=12
    )

    plt.ylabel(
        'Frekuenca',
        fontsize=12
    )

    plt.legend()

    plt.grid(
        True,
        linestyle=':',
        alpha=0.6
    )

    plt.tight_layout()

    plt.savefig(
        'rezultatet_rajmonda/xgboost_residuals_histogram.png',
        dpi=300
    )

    plt.close()

    # ==========================================
    # VIZUALIZIMI 4: SHAP Summary Plot
    # ==========================================
    print(" -> Duke gjeneruar SHAP Summary Plot...")

    explainer = shap.TreeExplainer(xg_model)

    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.title(
        'Ndikimi i Faktorëve në PM2.5 (SHAP)',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    plt.savefig(
        'rezultatet_rajmonda/xgboost_shap_summary.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # ==========================================
    # VIZUALIZIMI 5: Metrics Table
    # ==========================================
    print(" -> Duke gjeneruar Metrics Table...")

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ["Metrika", "Vlera"],
        ["R² Score", f"{r2:.4f}"],
        ["MAE", f"{mae:.4f} µg/m³"],
        ["RMSE", f"{rmse:.4f} µg/m³"]
    ]

    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.5, 0.5]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():

        if row == 0:
            cell.set_text_props(
                weight='bold',
                color='white'
            )

            cell.set_facecolor('#2c7bb6')

        else:
            cell.set_facecolor('#f9f9f9')

            if col == 0:
                cell.set_text_props(weight='bold')

    plt.title(
        'Performanca e Modelit XGBoost',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        'rezultatet_rajmonda/xgboost_metrics_table.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()
    print("\n -> Duke ruajtur rezultatet JSON për krahasimin përfundimtar...")
    MODEL_DIR = '../../Modelet'
    os.makedirs(MODEL_DIR, exist_ok=True)

    xgb_eval = {
        'model_name': 'XGBoost Regressor',
        'model_type': 'supervised',
        'author': 'Rajmonda',
        'target': 'pm2_5',
        'features_used': list(X.columns),
        'n_features': len(X.columns),
        'dataset_rows': len(df),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'split_method': 'random_split (80/20)',
        'metrics': {
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4),
            'R2': round(float(r2), 4),
        },
        'feature_importance': {
            row['Veçoria']: round(float(row['Rëndësia']), 4)
            for _, row in fi_df.iterrows()
        }
    }

    json_path = os.path.join(MODEL_DIR, 'xgboost_regressor.json')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(xgb_eval, f, indent=2, ensure_ascii=False)

    print(f"    Skedari JSON u ruajt me sukses te: {json_path}")

    print("\n==================================================")
    print("TË GJITHA VIZUALIZIMET U RUAJTËN ME SUKSES!")
    print("==================================================")

    return xg_model, y_test, y_pred


if __name__ == "__main__":

    skedari = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'

    trajner_xgboost(skedari)