from os.path import join

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA_DIR
from helpers import export_counter_speech_to_file
from llm_config import PROMPT

# Initialize model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda"  # the device to load the model onto
model_name = "mistral"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

print(f"Default temperature value: {model.generation_config.temperature}")  # results in 1.0
print(f"Default top_p value: {model.generation_config.top_p}")  # results in 1.0


# Function to generate counter speech
def generate_counter_speech(comment, prompt=PROMPT):
    # We define the system and user message
    system_prompt = prompt
    user_prompt = f"Generate counter speech to the following comment: {comment}."
    prefix = "Very concise and short counter speech that uses less than 200 tokens:"

    # Formats the messages according to Mistral's requirements
    messages = [
        {"role": "user", "content": f"{system_prompt} {user_prompt}"},
        {"role": "assistant", "content": prefix, "prefix": True}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = inputs.to(device)

    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=350,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    outputs = generated_ids[:, model_inputs.shape[-1]:]  # Cut off the original input length
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response[0]


if __name__ == '__main__':
    ct_df = pd.read_csv(join(DATA_DIR, "CTs_all.csv"))

    ct_df['counter_speech_mistral'] = ct_df['message'].apply(generate_counter_speech)
    export_counter_speech_to_file(ct_df, model_name)

    # Clean up resources to prevent memory leaks
    del model, tokenizer
    torch.cuda.empty_cache()
