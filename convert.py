#!/usr/bin/env python
import os
import torch
import re
import numpy as np
import string
import json
from tqdm import tqdm
import editdistance
from transformers import RobertaTokenizer, ElectraTokenizer


MATCH_IGNORE = {'do', 'did', 'does',
                'is', 'are', 'was', 'were', 'have', 'will', 'would',
                '?',}
PUNCT_WORDS = set(string.punctuation)
IGNORE_WORDS = MATCH_IGNORE | PUNCT_WORDS
MAX_LEN = 350
FILENAME = 'electra-large-discriminator'
FORCE=False
MODEL_FILE = '/home/electra-large-discriminator'
tokenizer = ElectraTokenizer.from_pretrained(MODEL_FILE, cache_dir=None)
DECISION_CLASSES = ['yes', 'no', 'more', 'irrelevant']
ENTAILMENT_CLASSES = ['yes', 'no', 'unknown']

tokenizer.add_special_tokens({'additional_special_tokens':["[RULE]"]})

def roberta_encode(doc):
    encoded = tokenizer.encode(doc.strip('\n').strip(), add_special_tokens=False)
    return encoded


def roberta_decode(doc):
    decoded = tokenizer.decode(doc, clean_up_tokenization_spaces=False).strip('\n').strip()
    return decoded


def load_data(filename):
    # print "Loading data:", filename
    # f_in = open(filename)
    # inp = f_in.readline()
    with open(filename) as f:
        data = json.loads(f)
    num_sent = 0
    cnt_multi_parents = 0
    for dialog in data:
        last_speaker = None
        turn = 0
        for edu in dialog["edus"]:
            edu["text_raw"] = edu["text"] + " "
            text = edu["text"]
            
            while text.find("http") >= 0:
                i = text.find("http")
                j = i
                while (j < len(text) and text[j] != ' '): j += 1
                text = text[:i] + " [url] " + text[j + 1:]
            
            invalid_chars = ["/", "\*", "^", ">", "<", "\$", "\|", "=", "@"]
            for ch in invalid_chars:
                text = re.sub(ch, "", text)
            tokens = []
            cur = 0
            for i in range(len(text)):
                if text[i] in "',?.!()\": ":
                    if (cur < i):
                        tokens.append(text[cur:i])
                    if text[i] != " ":
                        if len(tokens) == 0 or tokens[-1] != text[i]:
                            tokens.append(text[i])
                    cur = i + 1
            if cur < len(text):
                tokens.append(text[cur:])
            tokens = [token.lower() for token in tokens]
            for i, token in enumerate(tokens):
                if re.match("\d+", token): 
                    tokens[i] = "[num]"
            edu["tokens"] = tokens
            
            if edu["speaker"] != last_speaker:
                last_speaker = edu["speaker"]
                turn += 1
            edu["turn"] = turn
    return data

if __name__ == '__main__':
    with open('trees_mapping_electra-large-discriminator_dev.json') as f:
        data_raw = json.load(f)
    keys = list(data_raw.keys())    
    vals = [{'clause_t': [roberta_decode(t) for t in data_raw[k]['clause_t']], 'edu_t': [[roberta_decode(t) for t in i] for i in data_raw[k]['edu_t']]} for k in keys]

    data_raw = {k: v for k, v in zip(keys, vals)}
        
    keys = list(data_raw.keys())    
    mapping = []
    for k in keys:
        temp = {}
        temp['id'] = k
        temp['edus'] = []
        for clause_id, edus in enumerate(data_raw[k]['edu_t']):
            for edu_id, edu in enumerate(edus):
                temp['edus'].append({'text': edu, 'speaker': str(clause_id)})
        mapping.append(temp)

    num_sent = 0
    cnt_multi_parents = 0
    for dialog in mapping:
        last_speaker = None
        turn = 0
        for edu in dialog["edus"]:
            edu["text_raw"] = edu["text"] + " "
            text = edu["text"]
            
            while text.find("http") >= 0:
                i = text.find("http")
                j = i
                while (j < len(text) and text[j] != ' '): j += 1
                text = text[:i] + " [url] " + text[j + 1:]
            
            invalid_chars = ["/", "\*", "^", ">", "<", "\$", "\|", "=", "@"]
            for ch in invalid_chars:
                text = re.sub(ch, "", text)
            tokens = []
            cur = 0
            for i in range(len(text)):
                if text[i] in "',?.!()\": ":
                    if (cur < i):
                        tokens.append(text[cur:i])
                    if text[i] != " ":
                        if len(tokens) == 0 or tokens[-1] != text[i]:
                            tokens.append(text[i])
                    cur = i + 1
            if cur < len(text):
                tokens.append(text[cur:])
            tokens = [token.lower() for token in tokens]
            for i, token in enumerate(tokens):
                if re.match("\d+", token): 
                    tokens[i] = "[num]"
            edu["tokens"] = tokens
            
            if edu["speaker"] != last_speaker:
                last_speaker = edu["speaker"]
                turn += 1
            edu["turn"] = turn

    ftree = 'dev_for_train.json'
    with open(ftree, 'wt') as f:
        json.dump(mapping, f, indent=2)