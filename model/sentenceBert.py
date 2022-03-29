from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class SentenceBert(nn.Module):
    def __init__(self):
        super(SentenceBert, self).__init__()
        model_name = './pretraine/bert_base'
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        # how to set this hyper parameter 
        self.pooling = nn.MaxPool1d(3,stride=2)
        # self.linear = nn.Linear() 
        
    def forward(self,sent_a:List[str],sent_b:List[str]):
        batch_encoding_a: BatchEncoding = self.tokenizer(sent_a,padding=True)
        batch_encoding_b: BatchEncoding = self.tokenizer(sent_b,padding=True)
        sent_a_embedding = self.bert_model(torch.tensor(batch_encoding_a['input_ids']))
        sent_b_embedding = self.bert_model(torch.tensor(batch_encoding_b['input_ids']))
        pool_a = self.pooling(sent_a_embedding)
        pool_b = self.pooling(sent_b_embedding)
        concat_a_b = torch.cat((pool_a,pool_b,torch.abs(pool_a - pool_b)),dim=-1)
        linear = nn.Linear(concat_a_b.shape[-1],out_features=2)
        ln = linear(concat_a_b)
        res = F.softmax(ln)
        return res
