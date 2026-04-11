import pandas as pd
import os


def krijo_dataset_global(qytetet_kodet, input_dir, output_file):
    lista_df = []
    for qyteti, kodi in qytetet_kodet.items():
        skedari = os.path.join(input_dir, f"{qyteti}_ml_data.csv")

        if os.path.exists(skedari):
            df = pd.read_csv(skedari)

            df['qyteti'] = kodi
            lista_df.append(df)
            print(f" -> U lexua {qyteti.capitalize()} dhe u kodua me numrin {kodi}. (Rreshta: {len(df)})")
        else:
            print(f" [!] Skedari per {qyteti} nuk u gjet te {skedari}")

    if len(lista_df) > 0:
        df_global = pd.concat(lista_df, ignore_index=True)

        if 'sezoni_i_ngrohjes' in df_global.columns:
            df_global['sezoni_i_ngrohjes'] = df_global['sezoni_i_ngrohjes'].fillna(0).astype(int)
            print("\n -> 'Sezoni_i_ngrohjes' u rregullua.")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_global.to_csv(output_file, index=False)

        print(f"\nDataseti global u ruajt me sukses!")
        print(f"Total rreshta: {len(df_global)}")
        print(f"Skedari u ruajt te: {output_file}")
        print(f"Kolonat finale: {list(df_global.columns)}")
    else:
        print("\nNuk u gjet asnje dataset per tu bashkuar.")


if __name__ == "__main__":
    follderi_hyrje = '../../Datasetet/ml_ready_dataset/'
    skedari_dalje = '../../Datasetet/ml_ready_dataset/kosova_global_ml_data.csv'

    kodet = {
        'prishtine': 1,
        'prizren': 2,
        'peje': 3
    }

    krijo_dataset_global(kodet, follderi_hyrje, skedari_dalje)