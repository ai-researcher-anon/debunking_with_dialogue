import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from helpers import print_table
from llm_config import MODELS

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

from config import FINAL_ANNOTATION_FILE_PATH, ORDINAL_COLS, NOMINAL_COLS

[sys.path.append(i) for i in ['.', '..']]


def contingency_table(df, model_1, model_2, true_col='hate_speech_id_message'):
    true_labels = df[true_col]
    pred_1 = df[model_1]
    pred_2 = df[model_2]
    n_11 = n_10 = n_01 = n_00 = 0

    for true, pred1, pred2 in zip(true_labels, pred_1, pred_2):
        if pred1 == true and pred2 == true:
            n_11 += 1
        elif pred1 == true and pred2 != true:
            n_10 += 1
        elif pred1 != true and pred2 == true:
            n_01 += 1
        else:
            n_00 += 1

    return [[n_11, n_10], [n_01, n_00]]


def significance_test():
    results = []
    for category in ORDINAL_COLS:
        print(category)
        # Pivot data for this category
        pivot = df[["message", "model", category]].pivot(index="message", columns="model", values=category)

        valid_rows = pivot.dropna()
        if valid_rows.shape[0] > 0 and len(valid_rows.columns) >= 3:
            # Extract scores for Friedman test
            scores = [valid_rows[model] for model in valid_rows.columns]
            friedman_stat, friedman_p = friedmanchisquare(*scores)

            n, k = valid_rows.shape  # n = number of records, k = number of models
            kendall_w = friedman_stat / (n * (k - 1)) if n > 0 else None

            results.append((category, "Friedman", friedman_stat, friedman_p, kendall_w, valid_rows.shape[0]))

            if friedman_p < 0.05:
                comparisons = []
                models = pivot.columns
                for i, model_1 in enumerate(models):
                    for model_2 in models[i + 1:]:
                        # Select only rows where both models have values
                        paired_data = pivot[[model_1, model_2]].dropna()
                        if not paired_data.empty:
                            diff = paired_data[model_1] - paired_data[model_2]

                            if (diff == 0).all():
                                print(f"Models {model_1} and {model_2} perform identically for the selected metric.")
                                comparisons.append(
                                    (
                                        model_1, model_2, None, None, paired_data.shape[0],
                                        None))  # Record that no test was done
                                continue
                            w_stat, p = wilcoxon(paired_data[model_1], paired_data[model_2])

                            # Calculate Z-statistic
                            n_paired = paired_data.shape[0]
                            expected_w = n_paired * (n_paired + 1) / 4
                            std_w = np.sqrt(n_paired * (n_paired + 1) * (2 * n_paired + 1) / 24)
                            z_stat = (w_stat - expected_w) / std_w

                            # Calculate effect size
                            r = abs(z_stat / np.sqrt(n_paired))

                            if p < 0.05:  # If Wilcoxon is significant, determine the better model
                                better_model = model_1 if w_stat > 0 else model_2  # Based on the sign of the test statistic
                            else:
                                better_model = "No significant difference"

                            # Append results with effect size and better model
                            comparisons.append((model_1, model_2, z_stat, p, n_paired, r, better_model))

                # Apply multiple comparison correction only if pairwise comparisons are made
                if comparisons:
                    _, p_vals = zip(*[(stat, p) for _, _, stat, p, _, _, _ in comparisons])
                    corrected_pvals = multipletests(p_vals, method="bonferroni")[1]
                    for ((model_1, model_2, stat, _, support, r, better_model), corrected_p) in zip(comparisons,
                                                                                                    corrected_pvals):
                        results.append(
                            (category, f"{model_1} vs {model_2}", stat, corrected_p, r, support, better_model))

        else:
            print(f"  Skipping Friedman test for {category}: not enough valid data.")
            continue
        results_df = pd.DataFrame(results,
                                  columns=["Category", "Comparison", "Statistic", "P-value", "Effect Size", "Support",
                                           "Better Model"])
    return results_df


def import_data():
    df = pd.read_csv(FINAL_ANNOTATION_FILE_PATH)
    df.fillna("n/a", inplace=True)
    # Handle missing values
    for c in ORDINAL_COLS:
        df[c] = df[c].map(lambda x: np.nan if x is None or x == "n/a" or float(x) == 0.0 or pd.isnull(x) else float(x))
    for c in NOMINAL_COLS:
        try:
            df[c] = df[c].map(lambda x: np.nan if x is None or x == "n/a" or pd.isnull(x) else x)
        except:
            print(c)
            pass
    return df


if __name__ == '__main__':

    df = import_data()

    # Evaluate fear reactions
    print("Fear reaction stats")
    print(df.groupby("model")[["fear_id_response", "fear_id_message"]].agg(["mean", "median"]))

    print("Usage of CT and repetition of harmful content")
    df[["conspiracy_usage", "repetition_harmful"]] = df[["conspiracy_usage", "repetition_harmful"]].astype(float)
    print(df.groupby("model")[["conspiracy_usage", "repetition_harmful"]].agg(["mean", "median"]))

    print("Ordinal Statistics:")
    grouped = df.groupby("model")
    ordinal_stats = grouped[ORDINAL_COLS].agg(
        ["mean", "median", "std", "min", "max"]
    )
    print(ordinal_stats)
    stat_significance_test_results = significance_test()
    print("Results of statistical significance test of ordinal columns:")
    print_table(stat_significance_test_results, output_format="table")

    print("Nominal Statistics:")
    nominal_stats = {}
    for col in NOMINAL_COLS:
        nominal_stats[col] = grouped[col].value_counts()
    nominal_stats_df = pd.concat(nominal_stats, axis=1)
    print(nominal_stats_df)

    print("Hate speech classification metrics:")
    classification_metrics = {}
    df[['hate_speech_id_message', 'hate_speech_id_response']] = df[
        ['hate_speech_id_message', 'hate_speech_id_response']].astype(float)

    df_merged = df[df["model"] == MODELS[0]][['message', 'hate_speech_id_message', 'hate_speech_id_response']].rename(
        columns={'hate_speech_id_response': MODELS[0]}
    ).merge(
        df[df["model"] == MODELS[1]][['message', 'hate_speech_id_response']].rename(
            columns={'hate_speech_id_response': MODELS[1]}
        ),
        on='message',
        how='inner'
    ).merge(
        df[df["model"] == MODELS[2]][['message', 'hate_speech_id_response']].rename(
            columns={'hate_speech_id_response': MODELS[2]}
        ),
        on='message',
        how='inner'
    )

    print(len(df_merged))

    # df_merged has columns ['hate_speech_id_message', 'llama3', 'gpt4o', 'mistral']
    contingency_tables = {}

    for i, model1 in enumerate(MODELS):
        for model2 in MODELS[i + 1:]:
            table = contingency_table(df_merged, model1, model2)
            contingency_tables[f"{model1} vs {model2}"] = table
            print(f"Contingency Table for {model1} vs {model2}:")
            print(table)

    p_values = []
    for comparison, table in contingency_tables.items():
        result = mcnemar(table, exact=True)
        print(f"{comparison}: Statistic={result.statistic}, p-value={result.pvalue}")
        p_values.append(result.pvalue)

    # Adjust p-values for multiple comparisons
    corrected_pvals = multipletests(p_values, method="bonferroni")[1]
    print("Corrected p-values:", corrected_pvals)

    for model in MODELS:
        print(model)
        df_model = df[df["model"] == model].copy()
        print(classification_report(df_model['hate_speech_id_message'], df_model['hate_speech_id_response']))
        precision = precision_score(df_model['hate_speech_id_message'], df_model['hate_speech_id_response'])
        recall = recall_score(df_model['hate_speech_id_message'], df_model['hate_speech_id_response'])
        f1 = f1_score(df_model['hate_speech_id_message'], df_model['hate_speech_id_response'])
        classification_metrics[model] = {'Precision': precision, 'Recall': recall, 'F1': f1}

    metrics_df = pd.DataFrame(classification_metrics).T
    print(metrics_df)
