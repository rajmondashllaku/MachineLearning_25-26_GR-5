import pandas as pd
import numpy as np
import os


def pastro_te_dhenat(input_file, output_file, qyteti):
    print(f"\n--- Duke pastruar të dhënat për: {qyteti.capitalize()} ---")

    if not os.path.exists(input_file):
        print(f"Kujdes: Skedari {input_file} nuk u gjet. Po e kalojmë...")
        return

    # Leximi i datasetit
    df = pd.read_csv(input_file)
    rreshtat_fillestar = len(df)

    print("1. Largimi i vlerave null (Drop)...")
    # Zëvendësojmë tekstet e mundshme 'None' me vlerën e vërtetë null të numpy (NaN)
    df.replace('None', np.nan, inplace=True)

    # Sigurohemi që të gjitha kolonat (përveç 'time') të trajtohen si numra
    kolonat_numerike = df.columns.drop('time', errors='ignore')
    for col in kolonat_numerike:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fshijmë çdo rresht që ka qoftë edhe një vlerë të vetme null
    df = df.dropna()
    rreshtat_pas_null = len(df)
    print(f"   -> U fshinë {rreshtat_fillestar - rreshtat_pas_null} rreshta me vlera null.")

    print("2. Detektimi i Outliers (Vetëm raportim, PA I PREKUR)...")
    for col in kolonat_numerike:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        kufiri_poshtem = Q1 - 1.5 * IQR
        kufiri_siperm = Q3 + 1.5 * IQR

        # Thjesht numërojmë sa rreshta dalin jashtë normales statistikore
        outliers = df[(df[col] < kufiri_poshtem) | (df[col] > kufiri_siperm)]
        if len(outliers) > 0:
            print(f"   -> Kolona '{col}': U detektuan {len(outliers)} vlera potenciale anormale.")

    # Ruajtja e datasetit të pastruar
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Pastrimi për {qyteti.capitalize()} përfundoi! Dataseti u ruajt me vlerat e ndotjes të paprekura.")
    print(f"   Rreshtat përfundimtarë: {len(df)}")


if __name__ == "__main__":
    qytetet = ['prishtine', 'prizren', 'peje']

    for qyteti in qytetet:
        skedari_hyrje = f'../../Datasetet/processed_dataset/{qyteti}_integrated_data.csv'
        skedari_dalje = f'../../Datasetet/cleaned_dataset/{qyteti}_cleaned_data.csv'

        pastro_te_dhenat(skedari_hyrje, skedari_dalje, qyteti)