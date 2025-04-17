from os.path import join

import pandas as pd

from config import DATA_DIR, FINAL_ANNOTATION_FILE_PATH, COL_COUNTER_SPEECH_CORRECTED
from tabulate import tabulate


def extract_model_name_from_file(f):
    if "gpt" in f:
        return "gpt-4o"
    elif "llama" in f:
        return "llama"
    else:
        return "mistral"


def extract_corpus(model=None, theme=None, col=COL_COUNTER_SPEECH_CORRECTED):
    df = pd.read_csv(FINAL_ANNOTATION_FILE_PATH)
    df[col] = df[col].map(
        lambda x: x.strip(' "') if isinstance(x, str) else x).values.tolist()
    df.dropna(subset=[col], inplace=True)
    if theme:
        df = df[df["theme"] == theme]
    if model:
        df = df[df["model"] == model]
    return df[col].values.tolist()


def generate_random_corpus(random_state=42, theme=None, return_messages=False):
    """Generate corpus containing a random model response per message"""
    df = pd.read_csv(FINAL_ANNOTATION_FILE_PATH)
    if theme:
        df = df[df["theme"] == theme]
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # Group by 'message' and sample one random row per group
    random_responses = df.groupby('message').apply(lambda group: group.sample(1, random_state=random_state))

    random_responses.reset_index(drop=True, inplace=True)
    if return_messages:
        return random_responses["message"].values.tolist()
    return random_responses[COL_COUNTER_SPEECH_CORRECTED].values.tolist()


def export_counter_speech_to_file(df, model_name):
    df.to_csv(join(DATA_DIR, f'ct_counter_speech_{model_name}.csv'), index=False)
    df[df["theme"] == 1].to_csv(join(DATA_DIR, f'ct_theme_1_counter_speech_{model_name}.csv'), index=False)
    df[df["theme"] == 2].to_csv(join(DATA_DIR, f'ct_theme_2_counter_speech_{model_name}.csv'), index=False)


def print_table(results_df, output_format, file_path=None):
    """
    Prints a DataFrame in a specified format (LaTeX or table) and optionally writes it to a file.

    Args:
        results_df (pd.DataFrame): The DataFrame to print.
        output_format (str): The format to print ("latex" or "table").
        file_path (str, optional): The file path to save the table. If None, the table is not saved.
    """
    if output_format == "latex":
        table_string = results_df.to_latex(index=False, float_format="%.2f")
    elif output_format == "table":
        table_string = tabulate(results_df, headers='keys', tablefmt='grid', floatfmt=".2f")
    else:
        raise ValueError("Invalid output_format. Choose 'latex' or 'table'.")

    print(table_string)

    if file_path:
        results_df.to_csv(file_path)
        print(f"Table saved to {file_path}")
