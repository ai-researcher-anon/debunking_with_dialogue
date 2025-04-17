# Debunking with Dialogue

This repository provides an overview of the code and the data used as well as supplementary material of the paper *Debunking with Dialogue? Exploring AI-Generated Counterspeech to Challenge Conspiracy Theories*.

## Data

In `data`, you find: 

- annotated counterspeech results for the main experiment in `annotated_counterspeech.csv`

- annotated counterspeech results for the second experiment in `annotated_counterspeech_narrative.csv` 

- Likert scales used for annotation of counterspeech in `annotation_scales.pdf`

- The results for the diversity evaluation of the counterspeech responses by model as well as by model and theme

The original messages are available on request. 

## Code

In `evaluate`, you also find Python code as well as Jupyter notebooks for the evaluation  of the responses in terms of the criteria, diversity and NER. 

In `prepare_IAA_computation`, `prepare_annotation` and `prepare_counterspeech` you find the code to generate the counterspeech, as well as calculate the inter annotator agreement and prepare the files for annotation. 

## Annotation

In `likert_scales.pdf` you find the annotation criteria used.

## Generate your own counterspeech

To generate your own counterspeech, acquire credentials if you want to use GPT-4o, save them in the `.env`-file, create a virtual environment and install necessary packages, adjust the prompt in `llm_config.py` if needed, save your messages in a csv-file `data/CTs_all.csv` containing the column `message` and run `prepare_counterspeech/generate_counterspeech_<model>.py`. Results will be saved in `data`. 
