import numpy as np
import torch
from gensim import models
import torch.nn.functional as F
from udicOpenData.dictionary import *
from udicOpenData.stopwords import *

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__() # 繼承__init__功能
        self.hidden1 = torch.nn.Linear(n_features, n_hidden1) # 隱藏層線性輸出
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.output = torch.nn.Linear(n_hidden2, n_output) # 輸出層線性輸出

    def forward(self, x):
        # 正向傳播輸入值，神經網路分析出輸出值
        x = F.relu(self.hidden1(x)) # 激勵函數(隱藏層的線性值)
        x = F.relu(self.hidden2(x))
        x = F.sigmoid(self.output(x)) # 輸出值，但不是預測值，預測值還需另外計算
        return x

model = models.Word2Vec.load('/home/wmn/LinearNN/Word2vec_Wiki_zh/word2vec_wiki_zh.model.bin')
train = torch.load('net.pkl')

seg = list()
res = list()

while (True): 

    string = input('請輸入: ')
    seg = list(rmsw(string))
    # print(seg)
    for i in range(len(seg)):
        if seg[i] in model.wv.vocab:
            res.append(model[seg[i]])
    if res != 0:
        mean = np.mean(res, axis= 0)
    else:
        mean = np.zeros(400)
    x = torch.tensor(mean, dtype = torch.float)

    n = train(x).tolist()[0]
    p = train(x).tolist()[1]
    # print(train(x).tolist())
    if p > n:
        print('正面')
    else:
        print('負面')