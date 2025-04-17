import argparse
import os
import re
import sys
from collections import Counter
from os.path import join

[sys.path.append(i) for i in ['.', '..']]

import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import COL_COUNTER_SPEECH_CORRECTED, FINAL_ANNOTATION_FILE_PATH, DATA_DIR
from helpers import extract_corpus, generate_random_corpus, print_table

# nltk.download('punkt')
# nltk.download('punkt_tab')

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
SENTENCE_TRANSFORMER_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
from llm_config import MODELS


def distinct_n_score(responses, n=2):
    """measures the diversity of generated responses by calculating the ratio of
    unique n-grams (default bi-grams) to the total number of n-grams in the generated text.
    Higher values indicate more diversity."""
    ngrams = set()
    total_ngrams = 0
    for response in responses:
        tokens = response.split()
        response_ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        ngrams.update(response_ngrams)
        total_ngrams += len(response_ngrams)
    return len(ngrams) / total_ngrams if total_ngrams > 0 else 0


def calculate_self_bleu(corpus):
    """
    Calculate the Self-BLEU score for each response in the corpus by comparing it
    to the rest of the corpus.
    Returns:
        list of float: Self-BLEU scores for each response in the corpus.
    """
    smooth_fn = SmoothingFunction().method1

    self_bleu_scores = []

    for i, response in enumerate(corpus):
        generated_tokens = response.split()

        references = [ref.split() for j, ref in enumerate(corpus) if
                      j != i]  # excludes the response itself in comparisons

        score = sentence_bleu(references, generated_tokens, smoothing_function=smooth_fn)
        self_bleu_scores.append(score)

    return self_bleu_scores


def jaccard_similarity(response1, response2):
    """
    Computes the Jaccard similarity between two responses based on set overlap.
    """
    set1 = set(response1.split())
    set2 = set(response2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0


def calculate_diversity(responses):
    """
    Calculates the Diversity metric for each generated response in a set based on Jaccard similarity
    per response in comparison to the rest of the corpus.
    """
    diversity_scores = []
    for i, response in enumerate(responses):
        # Calculate Jaccard similarity with all other responses except the response itself
        max_jaccard = max(jaccard_similarity(response, responses[j])
                          for j in range(len(responses)) if j != i)
        diversity_score = 1 - max_jaccard
        diversity_scores.append(diversity_score)
    return diversity_scores


def simple_tokenize(sentence):
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", sentence)
    return tokens


def get_sentence_starts(texts, num_words=None, word_tokenizer=simple_tokenize):
    """
    Extracts the beginning of each sentence across multiple texts.
    """
    sentence_starts = []
    for text in texts:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = word_tokenizer(sentence)
            if num_words is None:
                sentence_starts.append(words)
            else:
                sentence_starts.append(words[:num_words])
    return sentence_starts


def distinct_n_sentence_starts(sentence_starts, n=2):
    ngrams = [tuple(start[:n]) for start in sentence_starts if len(start) >= n]
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0


def compute_semantic_similarity(texts, sim_model=SENTENCE_TRANSFORMER_MODEL):
    sim_scores = []
    for i, text in enumerate(texts):
        max_sim = max(cosine_similarity(sim_model.encode([text]), sim_model.encode([texts[j]]))
                      for j in range(len(texts)) if j != i)

        sim_scores.append(max_sim)
    return sim_scores


def compute_mean_similarities_by_model(df, sim_model=SENTENCE_TRANSFORMER_MODEL):
    """
    Computes mean semantic similarities between model pairs for each message
    and returns the results in a pivoted form where the model pairs are columns.

    Args:
        df (pd.DataFrame): DataFrame with columns 'message', 'model', COL_COUNTER_SPEECH_CORRECTED.

    Returns:
        pd.DataFrame: Pivoted DataFrame with mean similarities per model pair for each message.
    """
    results = []
    grouped = df.groupby('message')

    for message, group in grouped:
        responses = group[COL_COUNTER_SPEECH_CORRECTED].tolist()
        models = group['model'].tolist()
        embeddings = sim_model.encode(responses)
        similarity_matrix = cosine_similarity(embeddings)

        for i, model_1 in enumerate(models):
            for j, model_2 in enumerate(models):
                if i < j:  # Ensure unique pairs
                    mean_similarity = similarity_matrix[i, j]
                    results.append({
                        "message": message,
                        "model_pair": f"{model_1}_vs_{model_2}",
                        "mean_similarity": mean_similarity
                    })

    results_df = pd.DataFrame(results)

    # Pivot the DataFrame to make 'model_pair' columns and 'mean_similarity' as values
    pivoted_df = results_df.pivot(index='message', columns='model_pair', values='mean_similarity')

    return pivoted_df


def extract_emojis(input_string, only_exists=True):
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F700-\U0001F77F"  # Alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric symbols
        "\U0001F800-\U0001F8FF"  # Supplemental arrows
        "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
        "\U0001FA00-\U0001FA6F"  # Chess symbols, symbols of transport
        "\U0001FA70-\U0001FAFF"  # Symbols for healthcare, hands
        "\U00002600-\U000026FF"  # Miscellaneous symbols (e.g., ☀, ☔)
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F1E0-\U0001F1FF"  # Regional indicator symbols (flags)
        "]", flags=re.UNICODE
    )
    emojis = emoji_pattern.findall(input_string)
    if only_exists:
        return bool(emojis)
    return emojis


def extract_hashtags(input_string):
    """
    Detects whether a text contains hashtags and extracts them.
    Returns:
        tuple: A tuple (contains_hashtags, hashtags_list)
               where contains_hashtags is a boolean indicating if hashtags were found,
               and hashtags_list is a list of the hashtags.
    """
    hashtag_pattern = re.compile(r"#\w+")
    hashtags = hashtag_pattern.findall(input_string)
    contains_hashtags = len(hashtags) > 0

    return contains_hashtags, hashtags


def extract_urls(input_string):
    """
    Detects whether a text contains URLs and extracts them.
    Returns:
        tuple: A tuple (contains_urls, urls_list)
               where contains_urls is a boolean indicating if URLs were found,
               and urls_list is a list of the URLs.
    """
    url_pattern = re.compile(
        r"(https?://[^\s]+)"
    )
    urls = url_pattern.findall(input_string)
    contains_urls = len(urls) > 0

    return contains_urls, urls


def extract_named_entities(text):
    """
    Extracts named entities (persons and organizations) from a given text.
    Returns:
        dict: A dictionary with keys "persons" and "organizations", containing lists of extracted entities.
    """
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(text)

    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    contains_entities = bool(persons or organizations)
    return contains_entities, {"persons": persons, "organizations": organizations}


def compute_vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return 'positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate counter speech generated by LLMs")
    parser.add_argument("--by_theme", choices=['1', '0'], default="0", required=False,
                        help="Choose 1 if you want to differentiate by theme, 0 if you don't")
    parser.add_argument("--format", choices=["latex", "table"], default="table",
                        help="Output format: 'latex' for LaTeX table, 'table' for a pretty DataFrame table.")

    args = parser.parse_args()

    response_corpora = {}
    message_corpora = {}

    if args.by_theme == '1':
        for theme in [1, 2]:
            for model in MODELS:
                response_corpora[(theme, model)] = extract_corpus(model=model, theme=theme)
                message_corpora[(theme, model)] = extract_corpus(model=model, theme=theme, col="message")
            response_corpora[(theme, "random")] = generate_random_corpus(theme=theme)
            message_corpora[(theme, "random")] = generate_random_corpus(theme=theme, return_messages=True)
        results_output_file = "counterspeech_diversity_evaluation_by_model_theme.csv"

        # Initialize eval_df with a MultiIndex (model,theme) and create evaluation dataframe
        index = pd.MultiIndex.from_tuples(response_corpora.keys(), names=["theme", "model"])
        eval_df = pd.DataFrame(index=index)
    else:
        for model in MODELS:
            response_corpora[model] = extract_corpus(model=model, theme=None)
            message_corpora[model] = extract_corpus(model=model, theme=None, col="message")
        response_corpora["random"] = generate_random_corpus(theme=None)
        message_corpora["random"] = generate_random_corpus(theme=None, return_messages=True)

        # Initialize eval_df with a simple Index (model)
        index = pd.Index(response_corpora.keys(), name="model")
        eval_df = pd.DataFrame(index=index)
        results_output_file = "counterspeech_diversity_evaluation_by_model.csv"

    all_data = pd.read_csv(FINAL_ANNOTATION_FILE_PATH)
    print("Data loaded")
    # Corpus-level metrics
    # 1. Distinct n-grams
    eval_df['corpus_distinct_1grams'] = [distinct_n_score(response_corpora[key], n=1) for key in eval_df.index]
    eval_df['corpus_distinct_2grams'] = [distinct_n_score(response_corpora[key], n=2) for key in eval_df.index]
    print("Distinct n-grams calculated")

    # 2. Self-BLEU
    eval_df['corpus_self_bleu_mean'] = [np.mean(calculate_self_bleu(response_corpora[key])) for key in eval_df.index]
    eval_df['corpus_self_bleu_std'] = [np.std(calculate_self_bleu(response_corpora[key])) for key in eval_df.index]
    print("Self-BLEU calculated")

    # 3. Diversity based on Jaccard similarity
    eval_df['corpus_jaccard_diversity_mean'] = [np.mean(calculate_diversity(response_corpora[key])) for key in eval_df.index]
    eval_df['corpus_jaccard_diversity_std'] = [np.std(calculate_diversity(response_corpora[key])) for key in eval_df.index]
    print("Jaccard-diversity calculated")

    # 4. Distinct n-grams for sentence starts
    eval_df['corpus_distinct_3_sentence_starts'] = [distinct_n_sentence_starts(get_sentence_starts(response_corpora[key], 3))
                                                    for key in eval_df.index]
    eval_df['common_sentence_starts_3'] = [
        Counter([" ".join(item) for item in get_sentence_starts(response_corpora[key], 3)]).most_common(10) for key in
        eval_df.index]
    eval_df["common_sentences"] = [
        Counter([" ".join(item) for item in get_sentence_starts(response_corpora[key], None)]).most_common(10) for key in
        eval_df.index]
    print("Distinct sentence starts calculated")

    # # 5. Semantic similarity
    eval_df['corpus_semantic_similarity_mean'] = [np.mean(compute_semantic_similarity(response_corpora[key])) for key in
                                                  eval_df.index]
    eval_df['corpus_semantic_similarity_std'] = [np.std(compute_semantic_similarity(response_corpora[key])) for key in
                                                 eval_df.index]
    print("Semantic similarity calculated")

    # 5.b. Semantic similarity for messages as baseline
    unique_messages = all_data["message"].unique()
    sem_sim_messages = compute_semantic_similarity(unique_messages)
    print(f"Mean semantic similarity for messages: {np.mean(sem_sim_messages)}")
    print(f"Std semantic similarity for messages: {np.std(sem_sim_messages)}")
    messages_by_theme = all_data.groupby("theme")["message"].unique()
    for theme, messages in messages_by_theme.items():
        sem_sim_messages = compute_semantic_similarity(messages)
        print(f"Mean semantic similarity for messages in theme {theme}: {np.mean(sem_sim_messages)}")
        print(f"Std semantic similarity for messages in theme {theme}: {np.std(sem_sim_messages)}")

    # 6. Sentence structure: average sentence length, percentage of questions and exclamations, average text length
    sentence_corpus = [get_sentence_starts(response_corpora[key], None, word_tokenizer=word_tokenize) for key in eval_df.index]
    eval_df['mean_sentence_length'] = [np.mean([len(s) for s in sentences]) for sentences in sentence_corpus]
    eval_df['pct_questions'] = [
        sum([1 for text in item for s in text if s[-1] == "?"]) / sum(len(text) for text in item)
        for item in sentence_corpus
    ]
    eval_df['pct_exclamations'] = [
        sum([1 for text in item for s in text if s[-1] == "!"]) / sum(len(text) for text in item)
        for item in sentence_corpus
    ]
    eval_df['mean_text_length'] = [np.mean([len(text) for text in response_corpora[key]]) for key in eval_df.index]

    # Response-level metrics
    # 1. Distinct n-grams per response
    eval_df["response_distinct_2grams_mean"] = [np.mean([distinct_n_score([item], n=2) for item in response_corpora[key]])
                                                for key in eval_df.index]
    eval_df["response_distinct_2grams_std"] = [np.std([distinct_n_score([item], n=2) for item in response_corpora[key]]) for key
                                               in eval_df.index]
    print("Distinct n-grams per response calculated")

    # 7. (Number of) responses containing emojis
    eval_df["messages_with_emojis"] = [sum([extract_emojis(item) for item in response_corpora[key]]) for key in eval_df.index]
    eval_df["unique_emojis"] = [
        set([emoji for item in response_corpora[key] for emoji in extract_emojis(item, only_exists=False)]) for
        key in eval_df.index]
    print("Emojis calculated")

    # 8. (Number of) responses containing hashtags
    responses_with_hashtags = []
    all_hashtags = []

    for key in eval_df.index:
        # Extract hashtags for each message in the group
        hashtags_per_response = [extract_hashtags(item) for item in response_corpora[key]]

        # Count messages with at least one hashtag
        responses_with_hashtags.append(sum(1 for hashtags in hashtags_per_response if hashtags[0]))

        # Collect all hashtags (with repetitions)
        hashtags = [hashtag for _, tags in hashtags_per_response for hashtag in tags]
        all_hashtags.append(hashtags)

    # Update eval_df
    eval_df["responses_with_hashtags"] = responses_with_hashtags
    eval_df["hashtags"] = all_hashtags

    # 9. (Number of) responses containing URLs
    eval_df["responses_with_urls"] = [sum([extract_urls(item)[0] for item in response_corpora[key]]) for key in eval_df.index]
    eval_df["unique_urls"] = [set([url for item in response_corpora[key] for url in extract_urls(item)[1]]) for key in
                              eval_df.index]
    print("URLs calculated")

    # 10. Named entities
    messages_with_entities = []
    all_persons_response = []
    all_persons_message = []
    all_organizations_response = []
    all_organizations_message = []

    for key in eval_df.index:  # for each model or (model, theme) pair
        responses = response_corpora[key]
        messages = message_corpora[key]
        no_entities = 0
        persons_message, persons_response, organizations_message, organizations_response = [], [], [], []
        # Extract named entities for each message
        for i in range(len(responses)):
            ner_response = extract_named_entities(responses[i])
            ner_message = extract_named_entities(messages[i])

            persons_message.append(ner_message[1]["persons"])
            persons_response.append(ner_response[1]["persons"])
            organizations_message.append(ner_message[1]["organizations"])
            organizations_response.append(ner_response[1]["organizations"])
            if ner_response[0]:
                no_entities += 1  # Count responses with at least one entity
        all_persons_response.append(persons_response)
        all_persons_message.append(persons_message)
        all_organizations_response.append(organizations_response)
        all_organizations_message.append(organizations_message)
        messages_with_entities.append(no_entities)

    # Update eval_df
    eval_df["messages_with_entities"] = messages_with_entities
    eval_df["persons_message"] = all_persons_message
    eval_df["persons_response"] = all_persons_response
    eval_df["organizations_message"] = all_organizations_message
    eval_df["organizations_response"] = all_organizations_response

    print("Named entities calculated")

    # 11. Sentiment analysis
    eval_df["sentiment"] = [Counter([compute_vader_sentiment(item) for item in response_corpora[key]]) for key in eval_df.index]
    print("Sentiment analysis calculated")

    # 12. Compute mean pairwise similarity between responses per message
    pw_sim_responses = compute_mean_similarities_by_model(all_data)
    print("Mean pairwise similarity calculated")
    print(f"Results:{pw_sim_responses.describe()}")
    print_table(eval_df, args.format, file_path=join(DATA_DIR, results_output_file))
