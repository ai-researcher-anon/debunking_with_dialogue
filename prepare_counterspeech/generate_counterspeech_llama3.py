from os.path import join

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import DATA_DIR
from helpers import export_counter_speech_to_file
from llm_config import PROMPT

# Initialize model and tokenizer
model_name = "llama3"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to(device)


# Function to generate counter speech
def generate_counter_speech(comment, prompt=PROMPT):
    messages = [
        {"role": "system",
         "content": prompt},
        {"role": "user", "content": f"Generate counter speech to the following comment: {comment}"}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=350,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


if __name__ == '__main__':
    ct_df = pd.read_csv(join(DATA_DIR, "CTs_all.csv"))

    ct_df['counter_speech_llama3'] = ct_df['message'].apply(generate_counter_speech)

    export_counter_speech_to_file(ct_df, model_name)

    # Clean up resources to prevent memory leaks
    del model, tokenizer
    torch.cuda.empty_cache()
