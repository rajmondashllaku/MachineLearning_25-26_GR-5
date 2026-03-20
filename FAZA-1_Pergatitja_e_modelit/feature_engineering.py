import pandas as pd


def krijo_vecori_te_reja(input_file, output_file):
    print("---FEATURE ENGINEERING---")
    df = pd.read_csv(input_file)

    df['time'] = pd.to_datetime(df['time'])

    df['muaji'] = df['time'].dt.month
    df['ora'] = df['time'].dt.hour
    df['sezoni_ngrohjes'] = df['muaji'].isin([10, 11, 12, 1, 2, 3]).astype(int)

    df = df.drop(columns=['time'])

    df.to_csv(output_file, index=False)

    print("Dataset-i final eshte gati per trajnim.")
    print(f"E ruajtur tek: {output_file}")
    print(f"Kolonat finale: {list(df.columns)}\n")


if __name__ == "__main__":
    skedari_hyrje = '../Datasetet/processed_dataset/prishtina_cleaned_data.csv'
    skedari_dalje = '../Datasetet/processed_dataset/prishtina_ml_ready_data.csv'

    krijo_vecori_te_reja(skedari_hyrje, skedari_dalje)