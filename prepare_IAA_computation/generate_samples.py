from os.path import join

import pandas as pd

from config import DATA_DIR, annotation_cols

n_samples = 50

if __name__ == '__main__':
    df = pd.read_csv(join(DATA_DIR, "ct_counter_speech_all_processed.csv"))
    rel_cols = ["message", "model_strategies", "model_output", "model_counter_speech"]

    df[annotation_cols] = None  # add empty annotation columns
    df_sample = df[rel_cols + annotation_cols].sample(n_samples, random_state=42)
    df_sample.to_excel(join(DATA_DIR, "sampled_data.xlsx"))
