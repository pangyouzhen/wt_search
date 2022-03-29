# %%
import pandas as pd
df = pd.read_csv("./data/new.train.query.txt",sep="\t",names=["query"])


# %%
df[:5]

# %%
import torch
from torch import Tensor

# %%
from transformers import BertModel,BertTokenizer

# %%
from transformers import BatchEncoding


# %%
model_name = './model'
# 读取模型对应的tokenizer
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
# 载入模型
model: BertModel = BertModel.from_pretrained(model_name)

# %%
from typing import List, Dict, Tuple

# %%
input_text = "今天天气很好啊,你好吗"
# todo input_text2 怎样一起输入
input_text2 = "文本"
# 通过tokenizer把文本变成 token_id
batch_encoding: BatchEncoding = tokenizer([input_text,input_text2],padding=True)
print(f"{batch_encoding=}")
input_ids: List[int] = tokenizer.encode(input_text, add_special_tokens=True)
print(f"{len(input_ids)=}")
print(f"{input_ids=}")
# 101 代表cls 符号，102代表 sep
# [101, 791, 1921, 1921, 3698, 2523, 1962, 1557, 117, 872, 1962, 1408, 102]
input_ids: Tensor = torch.tensor([input_ids])
# 获得BERT模型最后一个隐层结果
print(input_ids.shape)
print(f"{batch_encoding['input_ids']=}")
with torch.no_grad():
    # bert 的输入为 batch_size,seq_length
    a = model(torch.tensor(batch_encoding['input_ids']))
    # a = model(input_ids)
    last_hidden_state = a.last_hidden_state
    # batch_size,seq_length,embedding
    print(torch.equal(last_hidden_state[:, 0, :], a.pooler_output))
    # last_hidden_state 的输出为 batch_size,seq_length, embedding
    print(last_hidden_state)
    print(last_hidden_state.shape)

# %%
import torch.nn as nn
m = nn.MaxPool1d(3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)

# %%
torch.cat((input,output),dim=-1).shape

# %%
output.shape

import inspect

print(inspect.signature(BertTokenizer.__init__))