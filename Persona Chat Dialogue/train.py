import numpy as numpy
from utils import get_data

# path = 'data/personachat/train_self_original.txt'
path = 'data/example_data.txt'
data = get_data(path)

for k in data[0].keys():
    print(k)