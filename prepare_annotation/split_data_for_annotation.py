from os.path import join

import pandas as pd

from config import DATA_DIR, annotation_col_mapping

if __name__ == '__main__':
    df = pd.read_csv(join(DATA_DIR, "ct_counter_speech_all_processed.csv"))
    rel_cols = ["message", "model_strategies", "model_output", "model_counter_speech"]
    ann_cols = list(annotation_col_mapping.values())
    df[ann_cols] = None  # add empty annotation columns
    df_sample = pd.read_excel(join(DATA_DIR, "sampled_data.xlsx"))
    df_rest = df[~df.index.isin(df_sample["Unnamed: 0"].values)]
    # split the data randomly into two parts
    df_rest_1 = df_rest.sample(frac=0.5, random_state=42)
    df_rest_2 = df_rest[~df_rest.index.isin(df_rest_1.index)]
    if len(df) == len(set(df_sample["Unnamed: 0"].values).union(set(df_rest_1.index)).union(set(df_rest_2.index))):
        print("Data split successfully")
        df_rest_1[rel_cols + ann_cols].to_excel(join(DATA_DIR, "annotation_data_2.xlsx"))
        df_rest_2[rel_cols + ann_cols].to_excel(join(DATA_DIR, "annotation_data_1.xlsx"))
