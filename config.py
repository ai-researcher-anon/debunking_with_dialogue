import os
from os.path import join, dirname

DIR_PATH = dirname(os.path.realpath(__file__))
DATA_DIR = join(DIR_PATH, "data")

# FINAL_ANNOTATION_FILE_PATH = join(DATA_DIR, "ct_counter_speech_all_annotations_merged.csv")
# The above file cannot be provided here as it contains the messages
# Thus, scripts requiring message content need to be adjusted be removing computation requiring the message content
FINAL_ANNOTATION_FILE_PATH = join(DATA_DIR, "annotated_counterspeech.csv")
annotation_cols = ["1 Conciseness and Clarity of Response", "2.1 Hate Speech Identification",
                   "2.2 If hate speech present (1-x): hate speech handling",
                   "2.3 If hate speech identified (x-1): conspiracy theory handling",
                   "3 Strategy 1: Factual Accuracy", "4 Strategy 2: Exploration of Alternative Explanations",
                   "5 Strategy 3: Use of Narrative Storytelling",
                   "6 Strategy 4: Encouragement of Critical Thinking",
                   "7.1 Fear Identification", "7.2 Empathy/Acknowledgement of Fear and Anxiety",
                   "8 Usage of terms such as conspiracy theory",
                   "9 Repetition of false or harmful content from the message"]
annotation_col_mapping = {"1 Conciseness and Clarity of Response": "clarity",
                          "3 Strategy 1: Factual Accuracy": "factual",
                          "4 Strategy 2: Exploration of Alternative Explanations": "alternative",
                          "5 Strategy 3: Use of Narrative Storytelling": "narrative",
                          "6 Strategy 4: Encouragement of Critical Thinking": "critical_thinking",
                          "7.1 Fear Identification": "fear_id",
                          "7.2 Empathy/Acknowledgement of Fear and Anxiety": "fear_ack",
                          "8 Usage of terms such as conspiracy theory": "conspiracy_usage",
                          "9 Repetition of false or harmful content from the message": "repetition_harmful",
                          "2.1 Hate Speech Identification": "hate_speech_id",
                          "2.2 If hate speech present (1-x): hate speech handling": "hate_speech_handling",
                          "2.3 If hate speech identified (x-1): conspiracy theory handling": "conspiracy_handling"}
STRATEGY_COLS = [f"strategy_{i}" for i in range(1, 5)]
COL_OUTPUT = "model_output"
COL_STRATEGY = "model_strategies"
COL_COUNTER_SPEECH = "model_counter_speech"
COL_STRATEGY_CORRECTED = "model_strategies_corrected"
COL_COUNTER_SPEECH_CORRECTED = "model_counter_speech_corrected"
ORDINAL_COLS = ["clarity", "factual", "alternative", "narrative", "critical_thinking", "fear_ack", "repetition_harmful",
                "hate_speech_handling", "conspiracy_handling"]
NOMINAL_COLS = ["fear_id", "conspiracy_usage", "hate_speech_id", "fear_id_message", "fear_id_response",
                "hate_speech_id_message", "hate_speech_id_response"]
