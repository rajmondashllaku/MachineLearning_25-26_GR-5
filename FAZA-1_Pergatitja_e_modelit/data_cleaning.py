import pandas as pd

def largo_outliers_iqr(df, kolonat, kufijte_mjedisor):
    df_pastruar = df.copy()

    for kolona in kolonat:
        Q1 = df_pastruar[kolona].quantile(0.25)
        Q3 = df_pastruar[kolona].quantile(0.75)
        IQR = Q3 - Q1

        kufiri_siperm_iqr = Q3 + (3.0 * IQR)

        kufiri_minimal_i_lejuar = kufijte_mjedisor[kolona]
        kufiri_siperm = max(kufiri_siperm_iqr, kufiri_minimal_i_lejuar)

        kufiri_poshtem = 0

        kushti = (df_pastruar[kolona] >= kufiri_poshtem) & (df_pastruar[kolona] <= kufiri_siperm)
        rreshtat_para = len(df_pastruar)
        df_pastruar = df_pastruar[kushti]
        rreshtat_pas = len(df_pastruar)

        print(
            f" -> [{kolona}] U fshine {rreshtat_para - rreshtat_pas} outliers (Kufiri i siperm i aplikuar: {kufiri_siperm:.2f})")

    return df_pastruar


def pastro_te_dhenat(input_file, output_file):
    print("--- Pastrimi i te dhenave ---")
    df = pd.read_csv(input_file)
    print(f"Madhesia e dataset-it PARA pastrimit: {len(df)} rreshta")

    df_clean = df.dropna(subset=['pm25', 'pm10'])

    df_clean = df_clean[(df_clean['pm25'] >= 0) & (df_clean['pm10'] >= 0)]

    print("\nDuke aplikuar metoden IQR Hibride per detektimin e anomalive...")

    limitet_reale = {
        'pm25': 300.0,
        'pm10': 500.0
    }

    df_clean = largo_outliers_iqr(df_clean, kolonat=['pm25', 'pm10'], kufijte_mjedisor=limitet_reale)

    print(f"\nMadhesia e dataset-it PAS pastrimit: {len(df_clean)} rreshta")

    df_clean.to_csv(output_file, index=False)
    print(f"Te dhenat u pastruan dhe u ruajten me sukses tek: {output_file}\n")


if __name__ == "__main__":
    skedari_hyrje = '../Datasetet/processed_dataset/prishtina_integrated_data.csv'
    skedari_dalje = '../Datasetet/processed_dataset/prishtina_cleaned_data.csv'

    pastro_te_dhenat(skedari_hyrje, skedari_dalje)
