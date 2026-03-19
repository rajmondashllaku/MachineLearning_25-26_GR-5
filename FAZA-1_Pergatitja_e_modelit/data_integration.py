import pandas as pd


def integro_motin_me_nje_vlere_reale(file_moti, file_ajri):
    print("1. Lexojme datasetet.")
    # Ndryshimi këtu: Përdorim variablat e funksionit
    moti_df = pd.read_csv(file_moti)
    ajri_df = pd.read_csv(file_ajri)

    print("2. Marrim vetem kolonat e kualitetit te ajrit dhe motit.")
    ajri_df = ajri_df[['period.datetimeFrom.utc', 'parameter.name', 'value']]
    ajri_df = ajri_df.rename(columns={'period.datetimeFrom.utc': 'time'})

    ajri_df = ajri_df[ajri_df['value'] != -999.0]

    print("3. Sinkronizojmë formatet e kohës (ZGJIDHJA).")
    # I kthejmë në format kohe, i rrumbullakosim në Orë ('h') dhe fshijmë Timezone
    moti_df['time'] = pd.to_datetime(moti_df['time'], utc=True).dt.round('h').dt.tz_localize(None)
    ajri_df['time'] = pd.to_datetime(ajri_df['time'], utc=True).dt.round('h').dt.tz_localize(None)

    print("4. Mbajmë VETËM 1 vlerë reale për çdo orë.")
    ajri_df = ajri_df.drop_duplicates(subset=['time', 'parameter.name'], keep='first')

    print("5. Përgatisim formatin e ri (kolonat pm10 dhe pm25).")
    ajri_pivot = ajri_df.pivot(index='time', columns='parameter.name', values='value').reset_index()
    ajri_pivot.columns.name = None

    print("6. Bashkojmë në një dataset të vetëm.")
    df_perfundimtare = pd.merge(moti_df, ajri_pivot, on='time', how='left')

    output_name = '../Datasetet/processed_dataset/prishtina_integrated_data.csv'
    df_perfundimtare.to_csv(output_name, index=False)

    print("\n--- INTEGRIMI PËRFUNDOI ME SUKSES ---")
    print(f"Numri i rreshtave në fund: {len(df_perfundimtare)}")
    print(f"E ruajtur si: {output_name}")


if __name__ == "__main__":
    # Rrugët ku ndodhen të dhënat e tua
    skedari_moti = '../Datasetet/unprocessed_datasets/prishtina_weather_raw_2020_now.csv'
    skedari_ajri = '../Datasetet/unprocessed_datasets/prishtina_air_quality.csv'

    # Thërrasim funksionin
    integro_motin_me_nje_vlere_reale(skedari_moti, skedari_ajri)