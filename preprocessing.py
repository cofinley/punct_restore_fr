import re
import os
import string
import math

import pandas as pd
import numpy as np
from tqdm import tqdm

MIN_SENTENCE_LENGTH = 7
MAX_SENTENCE_LENGTH = 70
MAX_SENTENCES = 100_000

token_col_name = 'tokens'
label_col_name = 'ner_tags'
b_sent = 'B-SENT'
other = 'O'

config = {
    "path_to_data": os.path.join(os.getcwd(), "data", "fr.txt"),
    "path_to_save": os.path.join(os.getcwd()),
    "train_size": 0.8,
    "sentence_batch_size": 64
}

END_PUNCTUATION = re.compile('[\.\!\?]$')
SPACE_AFTER_APOSTROPHE = re.compile("(\w')\s(\w+)")

def save_output(data: pd.DataFrame, path, filename):
    path = os.path.join(path, filename+".json")
    with open(path, 'w', encoding='utf-8') as file:
        data.to_json(file, orient='records', force_ascii=False, lines=True)

def get_df():
    all_tokens = []
    all_ner_tags = []
    i = 0
    with open(config["path_to_data"], encoding='utf-8') as f:
        for _, sentence in tqdm(enumerate(f), total=MAX_SENTENCES):
            if i == MAX_SENTENCES: break
            if sentence not in string.punctuation:
                sentence = sentence.lower().rstrip()
                sentence = re.sub(SPACE_AFTER_APOSTROPHE, r'\1\2', sentence)
                tokens = [re.sub(END_PUNCTUATION, '', word) for word in sentence.split() if word not in string.punctuation]
                len_tokens = len(tokens)
                # Important for building a good model: filter out small and large sentences
                if MIN_SENTENCE_LENGTH <= len_tokens <= MAX_SENTENCE_LENGTH:
                    i += 1
                    ner_tags = [b_sent, *[other for _ in range(len_tokens-1)]]
                    all_tokens.extend(tokens)
                    all_ner_tags.extend(ner_tags)

    df = pd.DataFrame()
    batch_tokens = []
    batch_ner_tags = []
    for j, token in tqdm(enumerate(all_tokens), total=len(all_tokens)):
        if len(batch_tokens) == config['sentence_batch_size']:
            df = df.append({'tokens': batch_tokens, 'ner_tags': batch_ner_tags}, ignore_index=True)
            batch_ner_tags = []
            batch_tokens = []
        ner_tag = all_ner_tags[j]
        batch_tokens.append(token)
        batch_ner_tags.append(ner_tag)
    else:
        print('left over batch:', len(batch_tokens))
        df = df.append({'tokens': batch_tokens, 'ner_tags': batch_ner_tags}, ignore_index=True)
    return df

def main():
    df = get_df()
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    train_size = config['train_size']
    validation_size = (1-train_size) / 2
    total_size = len(df)
    train_index = math.floor(total_size*train_size)
    validation_index = train_index + math.ceil(total_size*validation_size)
    df_train = df[:train_index]
    df_validation = df[train_index:validation_index]
    df_test = df[validation_index:]
    save_output(df_train, config["path_to_save"], "train")
    save_output(df_validation, config["path_to_save"], "validation")
    save_output(df_test, config["path_to_save"], "test")

main()