import pandas as pd
import os


def vlereso_datasetin_global(file_global):
    print("==================================================")
    print("RAPORTI I VLERËSIMIT")
    print("==================================================")

    if not os.path.exists(file_global):
        print(f"Skedari {file_global} nuk u gjet.\n")
        return

    df = pd.read_csv(file_global)

    print("1. Dimensionet e datasetit:")
    print(f"Total: {df.shape[0]} rreshta, {df.shape[1]} kolona\n")

    print("2. Vlerat Null (Duhet të jenë 0 nëse gjithçka shkoi mirë):")
    print(f"Total Nulls: {df.isnull().sum().sum()}\n")

    print("3. Shpërndarja e të dhënave sipas qyteteve:")
    shp_qyteteve = df['qyteti'].value_counts().sort_index()
    qytetet_emrat = {1: 'Prishtinë', 2: 'Prizren', 3: 'Pejë'}
    for kodi, nr_rreshta in shp_qyteteve.items():
        print(f"  {qytetet_emrat.get(kodi, kodi)} (Kodi {kodi}): {nr_rreshta} rreshta")
    print("\n")

    if 'sezoni_i_ngrohjes' in df.columns:
        print("4. Kontrolli i Sezonit të Ngrohjes (0 vs 1):")
        # Krijon një tabelë të vogël për të parë sa 1 dhe 0 ka secili qytet
        tabela_ngrohjes = df.groupby('qyteti')['sezoni_i_ngrohjes'].value_counts().unstack(fill_value=0)
        tabela_ngrohjes.index = tabela_ngrohjes.index.map(qytetet_emrat)
        print(tabela_ngrohjes)
        print("\n")

    print("5. Tipet e të dhënave (Të gjitha duhet të jenë numra):")
    tipet = pd.DataFrame(df.dtypes, columns=['Tipi']).reset_index()
    tipet.columns = ['Kolona', 'Tipi']
    print(tipet.to_string(index=False))
    print("\n")

    print("6. Outliers (Majat ekstreme të ruajtura):")
    if 'pm10' in df.columns:
        print(f"  PM10  -> Min: {df['pm10'].min()}, Max: {df['pm10'].max()}")
    if 'pm2_5' in df.columns:
        print(f"  PM2.5 -> Min: {df['pm2_5'].min()}, Max: {df['pm2_5'].max()}")
    print("\n==================================================")


if __name__ == "__main__":
    skedari_global = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'

    vlereso_datasetin_global(skedari_global)