import re
import pickle
import time
import os
import numpy as np

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
    return re.sub("[^\w]", " ",  sent).split()

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
                            training_example["user_message"] = [user_messages[i]]
                            training_example["model_response_candidates"] = model_response_candidates[i]
                            training_example["model_response"] = [model_response_targets[i]]
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
            training_example["user_message"] = [user_messages[i]]
            training_example["model_response_candidates"] = model_response_candidates[i]
            training_example["model_response"] = [model_response_targets[i]]
            training_example["model_persona"] = model_persona[:]
            train_data.append(training_example)
    return train_data

def tokenize_data(train_data):
    for i in range(len(train_data)):
        for j in range(len(train_data[i]["user_message"])):
            train_data[i]["user_message"][j] = tokenize_sent(train_data[i]["user_message"][j])
        for j in range(len(train_data[i]["model_response_candidates"])):
            train_data[i]["model_response_candidates"][j] = tokenize_sent(train_data[i]["model_response_candidates"][j])
        for j in range(len(train_data[i]["model_response"])):
            train_data[i]["model_response"][j] = tokenize_sent(train_data[i]["model_response"][j])
        for j in range(len(train_data[i]["model_persona"])):
            train_data[i]["model_persona"][j] = tokenize_sent(train_data[i]["model_persona"][j])
    if(path != 'data/example_data.txt'):
        pickle.dump(train_data, open('data/pickles/train_data.pkl', 'wb'))
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
    if(path != 'data/example_data.txt'):
        pickle.dump(vocab, open('data/pickles/vocab.pkl', 'wb'))
        pickle.dump(freq, open('data/pickles/freq.pkl', 'wb'))
    return vocab, freq, w2i

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

    # max_mem_size = max(max_mem_size, len(user_messages))
    # max_mem_size = max(max_mem_size, len(model_response_candidates))
    # max_mem_size = max(max_mem_size, len(model_response_targets))
    # max_mem_size = max(max_mem_size, len(model_persona))

def vectorize(data, max_mem_len, max_mem_size, w2i):
    train_data = []
    for i in range(len(data)):
        example = {}
        for k in data[i].keys():
            mem = []
            for sent in data[i][k]:
                sent = [w2i[word] for word in sent]
                mem.append(sent)
                sent += [0] * (max_mem_len - len(sent))
            while len(mem) < max_mem_size:
                mem.append([0] * max_mem_len)
            example[k] = mem
            mem = np.array(mem)
        train_data.append(example)
    pickle.dump(train_data, open('data/pickles/train_data_vectorized.pkl', 'wb'))
    return train_data

def get_data(path):
    max_mem_len = 38
    max_mem_size = 25
    if(path == 'data/example_data.txt'):
        train_data = read_data(path)
        train_data = tokenize_data(train_data)
        vocab, freq, w2i = build_vocab(path)
        train_data = vectorize(train_data, max_mem_len, max_mem_size, w2i)
    else:
        if os.path.exists('data/pickles/train_data_vectorized.pkl'):
            train_data = pickle.load(open('data/pickles/train_data_vectorized.pkl', 'rb'))
        else:
            if os.path.exists('data/pickles/train_data.pkl'):
                train_data = pickle.load(open('data/pickles/train_data.pkl', 'rb'))
            else:
                train_data = read_data(path)
                train_data = tokenize_data(train_data)

            if os.path.exists('data/pickles/vocab.pkl') and os.path.exists('data/pickles/freq.pkl'):
                vocab = pickle.load(open('data/pickles/vocab.pkl', 'rb'))
                freq = pickle.load(open('data/pickles/freq.pkl', 'rb'))
            else:
                vocab, freq, w2i = build_vocab(path)
            train_data = vectorize(train_data, max_mem_len, max_mem_size, w2i)
    return train_data

# path = 'data/personachat/train_self_original.txt'
path = 'data/example_data.txt'
time1 = time.time()
train_data = get_data(path)
# print(train_data)
# # print("time taken to load data is {} seconds".format(time.time()-time1))

