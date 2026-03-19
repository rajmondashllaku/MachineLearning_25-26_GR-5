import pandas as pd

def vlereso_te_dhenat(file_para, file_pas):
    print("VLERËSIMI TË TË DHËNAVE (PARA VS PAS)")

    df_para = pd.read_csv(file_para)
    df_pas = pd.read_csv(file_pas)

    print("\n1. MADHËSIA E DATASETIT:")
    print(f"PARA Pastrimit: {df_para.shape[0]} rreshta, {df_para.shape[1]} kolona")
    print(f"PAS Përpunimit:  {df_pas.shape[0]} rreshta, {df_pas.shape[1]} kolona")
    print(f"-> U fshinë totalisht {df_para.shape[0] - df_pas.shape[0]} rreshta me anomali ose mungesa.")

    print("\n2. VLERAT E ZBRAZËTA (NULL):")
    nulls_para = df_para.isnull().sum().sum()
    nulls_pas = df_pas.isnull().sum().sum()
    print(f"PARA Pastrimit: {nulls_para} vlera Null total")
    print(f"PAS Përpunimit:  {nulls_pas} vlera Null total (Trajtuar përmes dropna)")

    print("\n3. TIPET E TË DHËNAVE (DATA TYPES) NË DATASETIN FINAL:")
    tipet = pd.DataFrame(df_pas.dtypes, columns=['Tipi']).reset_index()
    tipet.columns = ['Kolona', 'Tipi']
    print(tipet.to_string(index=False))

    print("\n4. PËRJASHTUESIT (OUTLIERS) PËR TARGETIN:")
    print("PARA Pastrimit (kishte vlera negative dhe anomali):")
    print(f"PM10 -> Min: {df_para['pm10'].min()}, Max: {df_para['pm10'].max()}")
    print(f"PM25 -> Min: {df_para['pm25'].min()}, Max: {df_para['pm25'].max()}")
    print("\nPAS Përpunimit (të dhëna të pastra dhe reale):")
    print(f"PM10 -> Min: {df_pas['pm10'].min()}, Max: {df_pas['pm10'].max()}")
    print(f"PM25 -> Min: {df_pas['pm25'].min()}, Max: {df_pas['pm25'].max()}")


if __name__ == "__main__":
    skedari_para = '../Datasetet/processed_dataset/prishtina_integrated_data.csv'
    skedari_pas = '../Datasetet/processed_dataset/prishtina_ml_ready_data.csv'

    vlereso_te_dhenat(skedari_para, skedari_pas)
