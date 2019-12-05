# 資料處理套件
import csv
import numpy as np

# pytorch 相關套件
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F # 激勵函數

# word2vec 斷詞套件
from gensim import models
from udicOpenData.dictionary import *
from udicOpenData.stopwords import *

# 讀檔
data = list()
labels = list()

with open ('waimai_10k_tw.csv') as file:
    reader = csv.reader(file)
    for index, line in enumerate(reader):
        if index == 0:
            continue
        label, review = line
        data.append(review)
        labels.append(int(label))
# print(len(data))
# print(len(labels))

# 找出0和1的界線
sep_index = labels.index(0)
# print(sep_index)

# 建立資料 => train/test/valid
NUM = 3000
train_data = data[:NUM] + data[sep_index:sep_index+NUM]
train_labels = labels[:NUM] + labels[sep_index:sep_index+NUM]

test_data = data[NUM:NUM+500] + data[sep_index+NUM:sep_index+NUM+500]
test_labels = labels[NUM:NUM+500] + labels[sep_index+NUM:sep_index+NUM+500]

valid_data = data[NUM+500:NUM+1000] + data[sep_index+NUM+500:sep_index+NUM+1000]
valid_labels = labels[NUM+500:NUM+1000] + labels[sep_index+NUM+500:sep_index+NUM+1000]

# print('train:', len(train_data))
# print('test:',len(test_data))
# print('valid:',len(valid_data))

# 載入word2vec模型
w2v_model = models.Word2Vec.load('/home/wmn/LinearNN/Word2vec_Wiki_zh/word2vec_wiki_zh.model.bin')

def getData(x, y):
    data = list()
    labels = list()
    dim = w2v_model.wv.vector_size

    for i in range(len(x)): # or y
        label = torch.tensor(y[i], dtype = torch.long)
        review_word_list = list(rmsw(x[i]))
        review_tensor = torch.tensor([0]*dim, dtype = torch.float)
        word_count = 0
        for word in review_word_list:
            if word in w2v_model.wv.vocab:
                review_tensor += torch.tensor(w2v_model.wv[word], dtype = torch.float)
                word_count += 1
        if word_count != 0:
            data.append(review_tensor/word_count)
            labels.append(label)
    return torch.stack(data), torch.stack(labels)

# 獲得train/test/valid的data(tensor)
x_train, y_train = getData(train_data, train_labels)
x_test, y_test = getData(test_data, test_labels)
x_valid, y_valid = getData(valid_data, valid_labels)
# print(type(x_train))

# 轉換成TensorDataSet
trainData = TensorDataset(x_train, y_train)
testData = TensorDataset(x_test, y_test)
validData = TensorDataset(x_valid, y_valid)
# print(type(trainData))

# 產生dataloader
train_dataloader = DataLoader(trainData, batch_size=30, shuffle=True)
test_dataloader = DataLoader(testData, batch_size=30, shuffle=True)
valid_dataloader = DataLoader(validData, batch_size=30, shuffle=True)
# print(train_dataloader)

# 定義網路架構
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

# 計算答對多少筆
def accuracy(logits, labels):
    outputs = np.argmax(logits, axis=1)
    num_correct = torch.eq(outputs, labels).sum().item()
    # num_correct = sum([1 for i in range(len(outputs)) if torch.eq(outputs[i], labels[i])])
    # print(logits)
    # print()
    # print(outputs) # axis=1
    # print()
    # print(np.argmax(logits, axis=0))
    # print()
    # print(np.argmax(logits))
    # input()
    return num_correct

# 網路資訊
net = Net(400, 200, 100, 2)
#print(net)

# 設定optimizer/loss function
LR = 0.005
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

# 主要訓練迴圈
for epoch in range(10):
    # training
    train_loss = 0.0
    trian_correct = 0
    net.train()
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        logits = net(inputs)
        trian_correct += accuracy(torch.tensor(logits, requires_grad=False), labels)

        loss = loss_func(logits, labels)
        train_loss += loss.item()

        optimizer.zero_grad() # 清空上一步的殘餘更新參數值
        loss.backward() # 誤差反向傳播，計算參數更新值
        optimizer.step() # 將參數更新值施加到net的parameters上

    # validation
    valid_loss = 0.0
    valid_correct = 0
    net.eval()
    for i, data in enumerate(valid_dataloader):
        with torch.no_grad():
            inputs, labels = data
            logits = net(inputs)
            loss = loss_func(logits, labels)
        valid_loss += loss.item()
        valid_correct += accuracy(logits, labels)
    
    print(
        'epoch:', epoch+1,
        'train_loss:', round(train_loss/len(train_dataloader), 4),
        'train_acc:', round(trian_correct/len(trainData), 4),
        'valid_loss:', round(valid_loss/len(valid_dataloader), 4),
        'valid_acc:', round(valid_correct/len(validData), 4)
    )

torch.save(net,'net.pkl')