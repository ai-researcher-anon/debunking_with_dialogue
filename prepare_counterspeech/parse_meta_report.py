from os.path import join

import pandas as pd

from config import DATA_DIR


def separate_counter_speech_and_strategies(text):
    """Separate the counter speech and the strategies from the model output using the <XXX> tag
    specified in the prompt"""
    text_split = text.split("<XXX>")
    try:
        counter_speech = text_split[1]
        rest = text_split[0] + " " + "\n".join(text_split[2:])
    except IndexError:  # if there is no <XXX> in the text
        counter_speech = text
        rest = ""
    return counter_speech, rest


if __name__ == '__main__':
    df = pd.read_csv(join(DATA_DIR, "ct_counter_speech_all.csv"))
    df[['model_counter_speech', 'model_strategies']] = df["model_output"].apply(
        separate_counter_speech_and_strategies).apply(
        pd.Series)

    df.to_csv(join(DATA_DIR, "ct_counter_speech_all_processed.csv"), index=False)
