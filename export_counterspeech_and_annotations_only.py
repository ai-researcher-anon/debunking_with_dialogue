import sys
from os.path import join

import pandas as pd

[sys.path.append(i) for i in ['.', '..']]
from config import DATA_DIR, annotation_col_mapping, FINAL_ANNOTATION_FILE_PATH

if __name__ == '__main__':
    rel_cols = ["model", "theme", "model_output", "model_strategies", "model_counter_speech"]
    ann_cols = list(annotation_col_mapping.values())
    # Main experiment
    df = pd.read_csv(FINAL_ANNOTATION_FILE_PATH)
    df[rel_cols + ann_cols].to_csv(join(DATA_DIR, "annotated_counterspeech.csv"), index=False)
    # Narrative experiment
    df = pd.read_csv(join(DATA_DIR, "annotation_ct_counter_speech_gpt4o_3.csv"), sep=";")
    df["model"] = "gpt4o"
    df["theme"] = 2
    df.rename(columns={"counter_speech_gpt4o": "model_output"}, inplace=True)
    ann_cols = ["clarity", "narrative", "conspiracy_usage", "repetition_harmful", "hate_speech", "hate_speech_handling",
                "conspiracy_handling"]
    df[rel_cols + ann_cols].to_csv(join(DATA_DIR, "annotated_counterspeech_narrative.csv"), index=False)
