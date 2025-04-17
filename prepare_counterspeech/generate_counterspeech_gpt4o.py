import os
from os.path import join

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from config import DATA_DIR
from helpers import export_counter_speech_to_file
from llm_config import PROMPT

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
model_name = "gpt4o"


def generate_counter_speech(comment, prompt=PROMPT):
    response = client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": prompt
             },
            {"role": "user", "content": f"Generate counter speech to the following comment: {comment}."}

        ],
        model="gpt-4o",
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    ct_df = pd.read_csv(join(DATA_DIR, "CTs_all.csv"))

    ct_df['counter_speech_gpt4o'] = ct_df['message'].apply(generate_counter_speech)
    export_counter_speech_to_file(ct_df, model_name)
