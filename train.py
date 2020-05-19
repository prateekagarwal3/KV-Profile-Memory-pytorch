import numpy as np
from utils import get_data
from model import KVMemNN
from torch.autograd import Variable
import torch
import re
import pickle
import time
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

path = 'train_self_original.txt'
# path = 'example_data.txt's
train_data, mem_len, mem_size, vocab_size, w2i, i2w, vocab = get_data(path)
# for i in range(len(train_data)):
# 	print(train_data[i]["user_message"])
print(train_data[65714])
# print(len(data))


# embd_size = 500

# key = []
# val = []
# cands = []
# persona = []
# for i in range(len(data)):
#     key.append(data[i]["user_message"])
#     val.append(data[i]["model_response"])
#     persona.append(data[i]["model_persona"])
#     cands.append(data[i]["model_response_candidates"])
# print(key)
# print(len(key))
# # print(val[0])
# q = []
# target_responses = []
# target = []
# for b in range(len(key)):
#   for i in range(len(key[b])):
#     if(sum(key[b][i]) == 0):
#       target_responses.append(val[b][i-1])
#       val[b][i-1] = [0]*len(val[b][i-1])
#       q.append(key[b][i-1])
#       break
#     # elif(i == 24 and sum(key[b][i]) != 0):
#     #   target_responses.append(val[b][i])
#     #   val[b][i] = [0]*len(val[b][i])
#     #   q.append(key[b][i])
# # print(len(key), np.array(target_responses).shape)
# for b in range(len(cands)):
#     for i in range(len(cands[b])):
#         if(sum(cands[b][i]) == 0):
#             cands[b] = cands[b][0:i]
#             break
# # print(len(key), np.array(target_responses).shape)
# print(len(target_responses))
# for b in range(len(cands)):
#     for i in range(len(cands[b])):
#         if(cands[b][i] == target_responses[b]):
#             target.append(i)
#             break
# print("target length:", len(target))
# # print(len(key), np.array(target_responses).shape)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)