import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

def trajner_xgboost(file_path):
    print("==================================================")
    print("XGBOOST REGRESSOR")
    print("==================================================")

    if not os.path.exists(file_path):
        print(f"[!] Gabim: Skedari nuk u gjet te {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f" ->  Dimensioni: {df.shape}")
    kolonat_per_fshirje = ['pm2_5', 'pm10', 'time', 'sezoni_i_ngrohjes']
    X = df.drop(columns=[col for col in kolonat_per_fshirje if col in df.columns])
    y = df['pm2_5']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f" -> Rreshta per Trajnim: {X_train.shape[0]}")
    print(f" -> Rreshta per Testim:  {X_test.shape[0]}\n")

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

    eval_set = [(X_train, y_train), (X_test, y_test)]
    xg_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    y_pred = xg_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nREZULTATET E MODELIT:")
    print(f" * MAE (Gabimi Mesatar Absolut): {mae:.2f} µg/m³")
    print(f" * RMSE (Gabimi Mesatar Katror): {rmse:.2f} µg/m³")
    print(f" * R² (Saktësia):                {r2:.4f}")

    folderi_imazheve = '../../images'
    os.makedirs(folderi_imazheve, exist_ok=True)
    print("\nVizualizimet analitike...")

    # Vizualizimi 1: Rëndësia e Veçorive (Feature Importance)
    feature_importances = xg_model.feature_importances_
    fi_df = pd.DataFrame({
        'Veçoria': X.columns,
        'Rëndësia': feature_importances
    }).sort_values(by='Rëndësia', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Rëndësia', y='Veçoria', data=fi_df, palette='viridis', hue='Veçoria', legend=False)
    plt.title('Rëndësia e Faktorëve në Parashikimin e PM2.5 (XGBoost)', fontsize=14, fontweight='bold')
    plt.xlabel('Pesha (Importance Score)', fontsize=12)
    plt.ylabel('Veçoria (Feature)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_feature_importance.png'), dpi=300)
    plt.close()

    # Vizualizimi 2: Vlerat Reale vs Parashikimet (Actual vs Predicted)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='#2c7bb6', label='Parashikimet')
    max_val = max(max(y_test), max(y_pred))
    plt.plot([0, max_val], [0, max_val], color='#d7191c', linestyle='--', linewidth=2, label='Vija e Përsosmërisë')
    plt.title('Saktësia e XGBoost: Vlerat Reale vs Parashikimet', fontsize=14, fontweight='bold')
    plt.xlabel('Vlerat Reale të PM2.5 (µg/m³)', fontsize=12)
    plt.ylabel('Parashikimet e Modelit (µg/m³)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_actual_vs_predicted.png'), dpi=300)
    plt.close()

    # Vizualizimi 3: Kurba e të Mësuarit (Learning Curve)
    rezultatet_eval = xg_model.evals_result()
    numri_pemeve = len(rezultatet_eval['validation_0']['rmse'])
    x_axis = range(0, numri_pemeve)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, rezultatet_eval['validation_0']['rmse'], label='Gabimi në Trajnim', linewidth=2)
    plt.plot(x_axis, rezultatet_eval['validation_1']['rmse'], label='Gabimi në Testim', linewidth=2)
    plt.title('Kurba e të Mësuarit (XGBoost Learning Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Numri i Pemëve të Vendimit', fontsize=12)
    plt.ylabel('Gabimi RMSE (µg/m³)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_learning_curve.png'), dpi=300)
    plt.close()

    print(f" -> Të 3 grafikët u ruajtën me sukses në folderin: {folderi_imazheve}")

    # Vizualizimi 4: Tabela e Metrikave si Imazh
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')

    tabela_data = [
        ['MAE (Gabimi Mesatar Absolut)', f'{mae:.2f} µg/m³'],
        ['RMSE (Gabimi Mesatar Katror)', f'{rmse:.2f} µg/m³'],
        ['R² (Saktësia)', f'{r2:.4f}']
    ]
    kolonat = ['Metrika Vlerësuese', 'Rezultati i Modelit XGBoost']

    table = ax.table(cellText=tabela_data, colLabels=kolonat, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2)  # Zmadhimi për lexueshmëri

    # Stilimi i tabelës
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#0c4da2')  # Koka blu
        else:
            if col == 0:
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor('#f2f2f2')  # Kolona e parë gri
            else:
                cell.set_text_props(weight='bold', color='#1a7f37')
                cell.set_facecolor('#e6fcf5')  # Rezultatet jeshile

    plt.title('Performanca e Modelit Përfundimtar', fontsize=14, fontweight='bold', color='#333333', y=1.05)
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_metrics_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Vizualizimi 5: Shpërndarja e Gabimeve (Residual Plot)
    mbetjet = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(mbetjet, kde=True, color='#9b59b6', bins=40)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Gabimi Zero')
    plt.title('Shpërndarja e Gabimeve të Parashikimit (Residuals)', fontsize=14, fontweight='bold')
    plt.xlabel('Diferenca (Reale - Parashikuar) µg/m³', fontsize=12)
    plt.ylabel('Frekuenca (Numri i Ditëve)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_residuals_histogram.png'), dpi=300)
    plt.close()

    # Vizualizimi 6: SHAP Summary Plot (Explainable AI)
    print("\nDuke gjeneruar analizën e thellë SHAP (mund të marrë pak sekonda)...")

    # SHAP TreeExplainer është krijuar posaçërisht për modele si XGBoost
    explainer = shap.TreeExplainer(xg_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    # Gjenerojmë grafikun
    shap.summary_plot(shap_values, X_test, show=False)

    plt.title('Si ndikon secili faktor në Ndotjen PM2.5 (SHAP Values)', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'xgboost_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Grafiku SHAP u ruajt me sukses!")

    from sklearn.tree import DecisionTreeRegressor, plot_tree

    # Vizualizimi 7: Struktura e një Peme Vendimi (Alternativa Scikit-Learn)
    print("\nDuke gjeneruar një pemë vendimi ilustruese (pa Graphviz)...")
    pema_ilustruese = DecisionTreeRegressor(max_depth=3, random_state=42)
    pema_ilustruese.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    plot_tree(pema_ilustruese,
              feature_names=X.columns,
              filled=True,
              rounded=True,
              fontsize=12)

    plt.title('Logjika Ilustruese e një Peme Vendimi (Si punon algoritmi bazë)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(folderi_imazheve, 'decision_tree_logic_sklearn.png'), dpi=300)
    plt.close()
    print(" -> Pema ilustruese u ruajt me sukses!")

    # Ruajtja e Modelit (JSON)
    print("\nDuke ruajtur modelin...")
    folderi_modeleve = '../../modelet'
    os.makedirs(folderi_modeleve, exist_ok=True)

    shtegu_modelit = os.path.join(folderi_modeleve, 'xgboost_rajmonda.json')
    xg_model.save_model(shtegu_modelit)
    print(f" -> Modeli u ruajt me sukses në: {shtegu_modelit}")

    return xg_model, y_test, y_pred


if __name__ == "__main__":
    skedari = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'
    trajner_xgboost(skedari)