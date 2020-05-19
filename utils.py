import re
import pickle
import time
import os
import copy
import numpy as np

def read_data(path):
    user_messages = []
    model_response_targets = []
    model_response_candidates = []
    model_persona = None
    train_data = []
    cands_are_replies = False
    max_mem_size = -100
    with open(path) as read:
        for line in read:
            line = line.strip().replace('\\n', '\n')
            if len(line) > 0:
                if line[0:2] == '1 ':
                    lines_have_ids = True
                    if len(user_messages) != 0:
                        for i in range(len(user_messages)):
                            training_example = {}
                            training_example["user_message"] = user_messages[0:i+1]
                            training_example["model_response_candidates"] = copy.deepcopy(model_response_candidates[i])
                            training_example["model_response"] = model_response_targets[0:i+1]
                            training_example["model_persona"] = model_persona[:]
                            train_data.append(training_example)
                        user_messages = []
                        model_response_targets = []
                        model_response_candidates = []
                        model_persona = None
                        cands_are_replies = False
                if '\t' in line and not cands_are_replies:
                    cands_are_replies = True
                    model_persona = parse_persona(model_response_targets)
                    model_response_targets = []
                if lines_have_ids:
                    space_idx = line.find(' ')
                    line = line[space_idx + 1 :]
                    if cands_are_replies:
                        sp = line.split('\t')
                        if len(sp) > 1 and sp[1] != '':
                            user_messages.append(sp[0])
                            model_response_targets.append(sp[1])
                            model_response_candidates.append(split_model_response_candidates(sp[3]))
                    else:
                        model_response_targets.append(line)
                else:
                    model_response_targets.append(line)
        for i in range(len(user_messages)):
            training_example = {}
            training_example["user_message"] = user_messages[0:i+1]
            training_example["model_response_candidates"] = copy.deepcopy(model_response_candidates[i])
            training_example["model_response"] = model_response_targets[0:i+1]
            training_example["model_persona"] = model_persona[:]
            train_data.append(training_example)
    return train_data

def parse_persona(profile):
    modified_profile = []
    for p in profile:
        p = p[13:]
        modified_profile.append(p)
    return modified_profile

def split_model_response_candidates(model_response_candidates):
    model_response_candidates = model_response_candidates.split('|')
    return model_response_candidates

def tokenize_sent(sent):
    # print(sent)
    return re.sub("[^\w]", " ",  sent).split()

def tokenize_data(train_data, path):
    for i in range(len(train_data)):
        for j in range(len(train_data[i]["user_message"])):
            train_data[i]["user_message"][j] = tokenize_sent(train_data[i]["user_message"][j])
        # print(train_data[i]["model_response_candidates"])
        for j in range(len(train_data[i]["model_response_candidates"])):
            # for k in range(len(train_data[i]["model_response_candidates"][j])):
                # print(i, j, k)
            train_data[i]["model_response_candidates"][j] = tokenize_sent(train_data[i]["model_response_candidates"][j])
        for j in range(len(train_data[i]["model_response"])):
            train_data[i]["model_response"][j] = tokenize_sent(train_data[i]["model_response"][j])
        for j in range(len(train_data[i]["model_persona"])):
            train_data[i]["model_persona"][j] = tokenize_sent(train_data[i]["model_persona"][j])
    if(path != 'example_data.txt'):
        pickle.dump(train_data, open('train_data.pkl', 'wb'))
    return train_data

def build_vocab(path):
    freq = {}
    with open(path) as f:
        for line in f:
            line = tokenize_sent(line)
            for i in range(len(line)):
                if line[i] in freq:
                    freq[line[i]] += 1 
                else:
                    freq[line[i]] = 1
    vocab = []
    for k in freq.keys():
        vocab.append(k)
    w2i = dict((w, i) for i, w in enumerate(vocab, 1))
    i2w = dict((i, w) for i, w in enumerate(vocab, 1))
    # if(path != 'example_data.txt'):
    #     pickle.dump(vocab, open('vocab.pkl', 'wb'))
    #     pickle.dump(freq, open('freq.pkl', 'wb'))
    return vocab, freq, w2i, i2w

def max_mem_calculations():
    max_user_message_len = max([len(x) for y in range(len(train_data)) for x in train_data[y]["user_message"]])
    print(max_user_message_len)

    max_model_response_len = max([len(x) for y in range(len(train_data)) for x in train_data[y]["model_response"]])
    print(max_model_response_len)

    max_model_response_cand_len = max([len(x) for y in range(len(train_data)) for x in train_data[y]["model_response_candidates"]])
    print(max_model_response_cand_len)

    max_model_persona_len = -100
    for y in range(len(train_data)):
        for x in train_data[y]["model_persona"]:
            max_model_persona_len = max(max_model_persona_len, len(x))
    print(max_model_persona_len)

def vectorize(data, max_mem_len, max_mem_size, w2i, path):
    train_data = []
    for i in range(len(data)):
        example = {}
        for k in data[i].keys():
            mem = []
            for sent in data[i][k]:
                # print(k)
                sent = [w2i[word] for word in sent]
                mem.append(sent)
                sent += [0] * (max_mem_len - len(sent))
            while len(mem) < max_mem_size:
                mem.append([0] * max_mem_len)
            example[k] = mem
            mem = np.array(mem)
        train_data.append(example)
    if(path != 'example_data.txt'):
        pickle.dump(train_data, open('train_data_vectorized.pkl', 'wb'))
    return train_data

def get_data(path):
    max_mem_len = 38
    max_mem_size = 25
    if(path == 'example_data.txt'):
        train_data = read_data(path)
        # train_data = tokenize_data(train_data, path)
        vocab, freq, w2i, i2w = build_vocab(path)
        # train_data = vectorize(train_data, max_mem_len, max_mem_size, w2i, path)
    else:
        if os.path.exists('train_data_vectorized.pkl'):
            train_data = pickle.load(open('train_data_vectorized.pkl', 'rb'))
            vocab, freq, w2i, i2w = build_vocab(path)
        else:
            if os.path.exists('train_data.pkl'):
                train_data = pickle.load(open('train_data.pkl', 'rb'))
            else:
                train_data = read_data(path)
                # train_data = tokenize_data(train_data, path)
            vocab, freq, w2i, i2w = build_vocab(path)
            # train_data = vectorize(train_data, max_mem_len, max_mem_size, w2i, path)
    return train_data, max_mem_len, max_mem_size, len(vocab), w2i, i2w, vocab
