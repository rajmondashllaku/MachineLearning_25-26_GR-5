import pandas as pd
import os


def integro_motin_me_nje_vlere_reale(file_moti, file_ajri, output_name, qyteti):
    print(f"\n--- Duke integruar te dhenat per: {qyteti.capitalize()} ---")

    if not os.path.exists(file_moti) or not os.path.exists(file_ajri):
        print(f"Kujdes: Skedaret per {qyteti} nuk u gjeten. Po e kalojme...")
        return

    print("1. Lexojme datasetet.")
    moti_df = pd.read_csv(file_moti)
    ajri_df = pd.read_csv(file_ajri)

    print("2. Sinkronizojme formatet e kohes.")
    if 'date' in moti_df.columns:
        moti_df = moti_df.rename(columns={'date': 'time'})
    if 'date' in ajri_df.columns:
        ajri_df = ajri_df.rename(columns={'date': 'time'})

    moti_df['time'] = pd.to_datetime(moti_df['time'], utc=True).dt.round('h').dt.tz_localize(None)
    ajri_df['time'] = pd.to_datetime(ajri_df['time'], utc=True).dt.round('h').dt.tz_localize(None)

    if 'parameter.name' in ajri_df.columns:
        print(" -> U detektua formati OpenAQ. Pivot...")
        ajri_df = ajri_df[ajri_df['value'] != -999.0]
        ajri_df = ajri_df.drop_duplicates(subset=['time', 'parameter.name'], keep='first')
        ajri_pivot = ajri_df.pivot(index='time', columns='parameter.name', values='value').reset_index()
        ajri_pivot.columns.name = None
    else:
        print(" -> U detektua formati satelitor.")
        ajri_pivot = ajri_df

    print("3. Bashkojme ne nje dataset te vetem.")
    df_perfundimtare = pd.merge(moti_df, ajri_pivot, on='time', how='inner')

    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    df_perfundimtare.to_csv(output_name, index=False)

    print(f"Integrimi per {qyteti.capitalize()} perfundoi me sukses!")
    print(f"   Numri i rreshtave në fund: {len(df_perfundimtare)}")


if __name__ == "__main__":

    qytetet = ['prishtine', 'prizren', 'peje']

    for qyteti in qytetet:
        skedari_moti = f'../Datasetet/unprocessed_datasets/{qyteti}_weather_raw.csv'
        skedari_ajri = f'../Datasetet/unprocessed_datasets/{qyteti}_air_quality_sat.csv'
        output_name = f'../Datasetet/processed_dataset/{qyteti}_integrated_data.csv'

        integro_motin_me_nje_vlere_reale(skedari_moti, skedari_ajri, output_name, qyteti)