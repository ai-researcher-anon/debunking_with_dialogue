import argparse
import sys
from collections import Counter

[sys.path.append(i) for i in ['.', '..']]
from typing import Literal

import numpy as np

from config import STRATEGY_COLS, COL_OUTPUT, COL_COUNTER_SPEECH, COL_STRATEGY_CORRECTED, \
    COL_COUNTER_SPEECH_CORRECTED, FINAL_ANNOTATION_FILE_PATH
from helpers import print_table
from llm_config import MODELS

import pandas as pd

pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)


def check_strategies_explained(df):
    """Check whether model has specified any information regarding strategies and/or step 2"""
    return 100 * len(df[df[COL_STRATEGY_CORRECTED].map(lambda x: isinstance(x, str) and len(x.strip()) > 0)]) / len(df)


def check_output_contains_separator(df, separator="<XXX>"):
    return 100 * df[COL_OUTPUT].str.contains(separator).sum() / len(df)


def compare_auto_and_manual(df, mode: Literal["binary", "mean"] = "binary"):
    df = df[df[COL_STRATEGY_CORRECTED].map(lambda x: isinstance(x, str) and len(x.strip(' "')) > 0)]
    if mode == "binary":
        return 100 * (np.sum(df[COL_COUNTER_SPEECH].fillna("") == df[COL_COUNTER_SPEECH_CORRECTED].fillna(""))) / len(
            df)
    elif mode == "mean":
        diff = df.apply(lambda row: (
                                        len(row[COL_COUNTER_SPEECH]) if isinstance(row[COL_COUNTER_SPEECH], str) else 0
                                    ) - (
                                        len(row[COL_COUNTER_SPEECH_CORRECTED]) if isinstance(
                                            row[COL_COUNTER_SPEECH_CORRECTED], str) else 0
                                    ), axis=1)
        return np.round(diff.mean(), 2), np.round(diff.std(), 2)
    else:
        raise ValueError("Mode must be 'binary' or 'mean'")


def get_model_strategies(df, relative=False):
    strategies = {}
    for col in STRATEGY_COLS:
        applied = np.sum(df[col] == 1)
        not_applied = np.sum(df[col] == -1)
        unknown = np.sum(df[col].isnull())

        if relative:
            strategies[col] = float(np.round(100*applied /(applied + not_applied), 2))
        else:
            strategies[col] = {"applied": int(applied)}#, "not_applied": int(not_applied), "unknown": int(unknown)}
    return strategies


def compare_model_and_real_strategies(df):
    df = df.dropna(subset=STRATEGY_COLS)
    print(len(df))
    real_strategies = ["factual", "alternative", "narrative", "critical_thinking"]
    for c in real_strategies:
        df.loc[:, c] = df.loc[:, c].map(
            lambda x: -1.0 if x is None or x == "n/a" or float(x) == 0.0 or pd.isnull(x) else 1.0)

    strategy_combinations = df[STRATEGY_COLS].apply(lambda row: tuple(row), axis=1)
    real_strategy_combinations = df[real_strategies].apply(lambda row: tuple(row), axis=1)

    agreements = [i for i in range(len(strategy_combinations)) if
                  strategy_combinations.values[i] == real_strategy_combinations.values[i]]
    return 100 * len(agreements) / len(strategy_combinations.values)


def get_frequent_strategy_combinations(df):
    strategy_combinations = df[STRATEGY_COLS].dropna().apply(lambda row: tuple(row), axis=1)
    return Counter(strategy_combinations).most_common(10)


def frequency_step_2_handling_explained(df):
    return 100 * sum(df["step_2"] == 1) / (sum(df["step_2"].notnull()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate whether models separated counterspeech from meta report.")
    parser.add_argument("--format", choices=["latex", "table"], default="table",
                        help="Output format: 'latex' for LaTeX table, 'table' for a pretty DataFrame table.")
    args = parser.parse_args()
    df = pd.read_csv(FINAL_ANNOTATION_FILE_PATH)
    for c in [COL_COUNTER_SPEECH, COL_COUNTER_SPEECH_CORRECTED, COL_STRATEGY_CORRECTED, COL_OUTPUT]:
        df[c] = df[c].map(
            lambda x: x.strip(' "') if isinstance(x, str) else x)
    df[STRATEGY_COLS] = df[STRATEGY_COLS].replace("", np.nan)

    results = []

    for model in ["all"] + MODELS:
        model_df = df if model == "all" else df[df["model"] == model]
        len_df = len(model_df)

        strategy_pct = check_strategies_explained(model_df)
        separator_pct = check_output_contains_separator(model_df)
        counter_speech_length_mean, counter_speech_length_std = compare_auto_and_manual(model_df, mode="mean")
        binary_comparison = compare_auto_and_manual(model_df, mode="binary")
        step_2_applied = frequency_step_2_handling_explained(model_df)
        strategies = get_model_strategies(model_df)
        strategies_rel = get_model_strategies(model_df, relative=True)
        frequent_combinations = get_frequent_strategy_combinations(model_df)
        percentage_agreement_real_model_strategy = compare_model_and_real_strategies(model_df)
        model_results = {
            "model": model,
            "strategies specified (%)": strategy_pct,
            "separator present (%)": separator_pct,
            "mean raw counterspeech length": np.round(model_df[COL_COUNTER_SPEECH].dropna().str.len().mean(), 2),
            "mean counterspeech length": np.round(
                model_df[COL_COUNTER_SPEECH_CORRECTED].dropna().str.len().mean(), 2),
            "mean_diff_counter_speech_length": counter_speech_length_mean,
            "std_diff_counter_speech_length": counter_speech_length_std,
            "counterspeech correctly separated (%)": binary_comparison,
            "step 2 specified (%)": step_2_applied,
            "specified strategies applied (%))": percentage_agreement_real_model_strategy,
            "strategies": strategies,
            "strategies_rel": strategies_rel,
            "frequent strategies": frequent_combinations
        }

        results.append(model_results)
    results_df = pd.DataFrame(results)
    results_df["corrected_vs_reported_length"] = 100 * results_df["mean counterspeech length"] / results_df[
        "mean raw counterspeech length"]

    print_table(results_df, args.format)
    paper_cols_1 = ["model",  "strategies specified (%)","counterspeech correctly separated (%)",
                    "step 2 specified (%)", "mean counterspeech length"]
    print_table(results_df[paper_cols_1], args.format)

