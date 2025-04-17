import argparse
from os.path import join

import krippendorff
import numpy as np
import pandas as pd

from config import DATA_DIR, annotation_col_mapping, ORDINAL_COLS, NOMINAL_COLS
from helpers import print_table

annotator_2_col_mapping = {"Clarity": "clarity",
                           "Factual Accuracy": "factual",
                           "Exploration of Alternative Explanations": "alternative",
                           "Narrative Storytelling": "narrative",
                           "Critical Thinking": "critical_thinking",
                           "Fear Identification": "fear_id",
                           "Empathy for Fear & Anxiety": "fear_ack",
                           "Term Usage (CT, misinfo)": "conspiracy_usage",
                           "Repetition of false content": "repetition_harmful",
                           "2.1 Hate Speech Identification": "hate_speech_id",
                           "2.2 If hate speech present (1-x): hate speech handling": "hate_speech_handling",
                           "2.3 If hate speech identified (x-1): conspiracy theory handling": "conspiracy_handling"}


def compute_value_domain(responses):
    all_values = set.union(*[set(row) for row in responses])
    return {val for val in all_values if not (isinstance(val, float) and np.isnan(val))}


def normalize_values(responses_1, responses_2, col_type):
    both = [responses_1, responses_2]
    if col_type == "ordinal":
        return [[np.nan if x is None or x == "n/a" or float(x) == 0.0 or pd.isnull(x) else x for x in row] for row in
                both]
    elif col_type == "nominal":
        return [[np.nan if x is None or x == "n/a" or pd.isnull(x) else x for x in row] for row in both]
    elif col_type == "binary":
        return [["0" if x is None or x == "n/a" or float(x) == 0.0 or pd.isnull(x) else "1" for x in row] for row in
                both]
    else:
        raise ValueError(f"Unknown column type {col_type}")


def generate_table(cols, df1, df2, col_type, alpha_type):
    results = []
    for col in cols:
        both = normalize_values(df1[col].values.tolist(), df2[col].values.tolist(), col_type)
        reliability_data = np.array(both, dtype=float) if col_type == "ordinal" else np.array(both, dtype=str)
        all_values = compute_value_domain(reliability_data)
        if len(all_values) < 2:
            print(f"Skipping {col} due to insufficient values")
            continue
        alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement=alpha_type,
                                   value_domain=list(all_values))
        results.append({"Column": col, "Alpha": alpha, "Values": [str(v) for v in all_values]})

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Krippendorff's alpha and output as LaTeX or table.")
    parser.add_argument("--format", choices=["latex", "table"], default="table",
                        help="Output format: 'latex' for LaTeX table, 'table' for a pretty DataFrame table.")
    args = parser.parse_args()

    df_1 = pd.read_excel(join(DATA_DIR, "sampled_data_1.xlsx"))
    df_2 = pd.read_excel(join(DATA_DIR, "sampled_data_2.xlsx"))
    df_1.fillna("n/a", inplace=True)

    df_1.rename(columns=annotation_col_mapping, inplace=True)
    df_2.rename(columns=annotator_2_col_mapping, inplace=True)

    for df in [df_1, df_2]:
        df[["fear_id_message", "fear_id_response"]] = df["fear_id"].str.split("-", expand=True)
        df[["hate_speech_id_message", "hate_speech_id_response"]] = df["hate_speech_id"].str.split("-", expand=True)

    print("ORDINAL COLUMNS")
    results_ordinal = generate_table(ORDINAL_COLS, df_1, df_2, "ordinal", alpha_type="ordinal")
    print_table(results_ordinal, args.format)

    print("NOMINAL COLUMNS:")
    results_nominal = generate_table(NOMINAL_COLS, df_1, df_2, "nominal", alpha_type="nominal")
    print_table(results_nominal, args.format)

    print("ORDINAL COLUMNS TREATED AS BINARY TO ACCOUNT FOR n/a:")
    results_binary = generate_table(ORDINAL_COLS, df_1, df_2, "binary", alpha_type="nominal")
    print_table(results_binary, args.format)
