import pandas as pd
import os


def krijo_features_profesionale(input_file, output_file, qyteti):
    print(f"\n--- Feature Engineering: {qyteti.capitalize()} ---")

    if not os.path.exists(input_file):
        print(f"Skedari {input_file} nuk u gjet.")
        return

    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])

    print("1. Ekstraktimi i veçorive baze...")
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek

    print("2. Domain Knowledge...")
    if qyteti == 'prishtine':
        print("   -> U shtua kolona 'sezoni_i_ngrohjes' (Prishtine).")
        df['sezoni_i_ngrohjes'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)

    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    print("3. Heqja e kolones tekstuale...")
    df = df.drop(columns=['time'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Feature Engineering per {qyteti.capitalize()} perfundoi me sukses!")
    print(f"Kolonat finale: {list(df.columns)}")


if __name__ == "__main__":
    qytetet = ['prishtine', 'prizren', 'peje']

    for qyteti in qytetet:
        skedari_hyrje = f'../Datasetet/cleaned_dataset/{qyteti}_cleaned_data.csv'
        skedari_dalje = f'../Datasetet/ml_ready_dataset/{qyteti}_ml_data.csv'

        krijo_features_profesionale(skedari_hyrje, skedari_dalje, qyteti)