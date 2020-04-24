import numpy as np
from utils import get_data
from model import KVMemNN
import torch

# path = 'data/personachat/train_self_original.txt'
path = 'data/example_data.txt'
data, mem_len, mem_size, vocab_size = get_data(path)
# print(data[3]["model_response_candidates"])
# print(mem_len, mem_size)
embd_size = 500

model = KVMemNN(mem_len, mem_size, embd_size, vocab_size)
q = None
for i in range(len(data[3]["user_message"])):
    if(sum(data[3]["user_message"][i]) == 0):
        q = torch.tensor(data[3]["user_message"][i-1]).reshape(1, -1)
        break

persona = torch.tensor(data[3]["model_persona"]).unsqueeze(0)
key = torch.tensor(data[3]["user_message"]).unsqueeze(0)
val = torch.tensor(data[3]["model_response"]).unsqueeze(0)
cands = torch.tensor(data[3]["model_response_candidates"]).unsqueeze(0)
for i in range(len(cands[0])):
    if(sum(cands[0][i]) == 0):
        cands = cands[0][0:i]
        break
model(q, persona, key, val, cands)