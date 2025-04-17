import sys
from os.path import join

[sys.path.append(i) for i in ['.', '..']]
import pandas as pd

from config import DATA_DIR, annotation_col_mapping, STRATEGY_COLS, COL_STRATEGY, COL_COUNTER_SPEECH, \
    COL_STRATEGY_CORRECTED, COL_COUNTER_SPEECH_CORRECTED, FINAL_ANNOTATION_FILE_PATH

original_output_file = "ct_counter_speech_all_processed.csv"
annotation_1 = "annotation_data_annotator_1.xlsx"
annotation_2 = "annotation_data_annotator_2.xlsx"
sampled_data = "sampled_data_annotator_1.xlsx"  # in this file the conflicts are resolved


def rename_model_output_cols(df):
    """Rename columns containing model strategies and counter speech so that
    automatically parsed and manually corrected versions can be differentiated in the merged dataframe"""
    return df.rename(columns={COL_COUNTER_SPEECH: COL_COUNTER_SPEECH_CORRECTED, COL_STRATEGY: COL_STRATEGY_CORRECTED})


if __name__ == '__main__':
    """Load files and rename model output columns"""
    df = pd.read_csv(join(DATA_DIR, original_output_file))
    df_1 = rename_model_output_cols(pd.read_excel(join(DATA_DIR, annotation_1)))
    df_2 = rename_model_output_cols(pd.read_excel(join(DATA_DIR, annotation_2)))
    df_sample = rename_model_output_cols(pd.read_excel(join(DATA_DIR, sampled_data)))

    # Check that "Unnamed: 0" values are distinct as they are used as index
    if not set(df_1["Unnamed: 0"]).isdisjoint(df_2["Unnamed: 0"]):
        raise ValueError("The values in 'Unnamed: 0' overlap between df_h and df_m.")

    "Merge dataframes"
    ann_cols = list(annotation_col_mapping.values())
    cols_annotated_files = ["Unnamed: 0", COL_STRATEGY_CORRECTED,
                            COL_COUNTER_SPEECH_CORRECTED] + ann_cols + STRATEGY_COLS + ["step_2"]
    print(set(df.columns).intersection(set(cols_annotated_files)))

    df_merged = df.copy()
    for df_annotated in [df_sample, df_2, df_1]:
        df_merged = df_merged.merge(df_annotated[cols_annotated_files], left_on=df.index, right_on="Unnamed: 0",
                                    how="left", suffixes=("", "_annotated"))
        for col in df_annotated.columns:
            if col in df_merged.columns and col + "_annotated" in df_merged.columns:
                df_merged[col] = df_merged[col + "_annotated"].combine_first(
                    df_merged[col])  # overwrite joint columns with values in annotated files
                df_merged.drop(columns=[col + "_annotated"], inplace=True)

    df_merged.drop("Unnamed: 0", axis=1, inplace=True)
    "Check if merge was successful"
    assert len(df_merged) == len(df), "Merge failed"
    # post-process columns
    df_merged["conspiracy_usage"] = df_merged["conspiracy_usage"].fillna("0")

    for c in ["fear_id", "hate_speech_id"]:
        df_merged[c] = df_merged[c].fillna("0-0")
        df_merged[[f"{c}_message", f"{c}_response"]] = df_merged[c].str.split("-", expand=True)
        df_merged[[f"{c}_message", f"{c}_response"]] = df_merged[[f"{c}_message", f"{c}_response"]].astype(float)

    print(df_merged.columns)
    df_merged.to_csv(FINAL_ANNOTATION_FILE_PATH, index=False)
    df_merged.to_excel(FINAL_ANNOTATION_FILE_PATH.replace("csv", "xlsx"))
