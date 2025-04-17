from os.path import join

import pandas as pd

from config import DATA_DIR
from llm_config import MODELS

all_data = []
for m in MODELS:
    df = pd.read_csv(join(DATA_DIR, f"ct_counter_speech_{m}.csv"))
    df["model"] = m
    counter_speech_col = [c for c in df.columns if "counter_speech" in c][0]
    df.rename(columns={counter_speech_col: "model_output"}, inplace=True)
    all_data.append(df)
all_data = pd.concat(all_data)
all_data.to_csv(join(DATA_DIR, "ct_counter_speech_all.csv"), index=False)
