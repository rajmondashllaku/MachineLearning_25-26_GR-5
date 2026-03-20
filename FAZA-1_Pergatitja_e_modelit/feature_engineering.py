import pandas as pd


def largo_outliers_manual_sezonal(df):
    print("\n-> Duke aplikuar pastrimin manual sipas stineve (Domain Knowledge)...")
    rreshtat_para = len(df)

    mask_vere = (df['sezoni_ngrohjes'] == 0) & (df['pm25'] <= 60.0) & (df['pm10'] <= 120.0)

    mask_dimer = (df['sezoni_ngrohjes'] == 1) & (df['pm25'] <= 200.0) & (df['pm10'] <= 250.0)

    df_pastruar = df[mask_vere | mask_dimer].copy()

    rreshtat_pas = len(df_pastruar)
    print(f"   U fshinë {rreshtat_para - rreshtat_pas} rreshta me anomali nga sensoret.")

    return df_pastruar

def krijo_vecori_te_reja(input_file, output_file):
    print("---FEATURE ENGINEERING---")
    df = pd.read_csv(input_file)

    df['time'] = pd.to_datetime(df['time'])

    df['muaji'] = df['time'].dt.month
    df['ora'] = df['time'].dt.hour
    df['sezoni_ngrohjes'] = df['muaji'].isin([10, 11, 12, 1, 2, 3]).astype(int)

    df = df.drop(columns=['time'])
    df = largo_outliers_manual_sezonal(df)



    df.to_csv(output_file, index=False)

    print("Dataset-i final eshte gati per trajnim.")
    print(f"E ruajtur tek: {output_file}")
    print(f"Kolonat finale: {list(df.columns)}\n")


if __name__ == "__main__":
    skedari_hyrje = '../Datasetet/processed_dataset/prishtina_cleaned_data.csv'
    skedari_dalje = '../Datasetet/processed_dataset/prishtina_ml_ready_data.csv'

    krijo_vecori_te_reja(skedari_hyrje, skedari_dalje)