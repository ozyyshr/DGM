import json, re
import numpy as np

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
    with open('converted_train_.json') as f:
        data = json.load(f)
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

    ftree = 'dev_for_train.json'
    with open(ftree, 'wt') as f:
        json.dump(data, f, indent=2)