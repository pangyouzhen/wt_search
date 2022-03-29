import torch 
import torch.nn as nn
import torch.nn.functional as F

class SentenceBert(nn.Module):
    def __init__(self):
        super(SentenceBert, self).__init__()
        self.bert_model = BertModel.from_pretrained("./model")
        # how to set this hyper parameter 
        self.pooling = nn.MaxPool1d(3,stride=2)
        # self.linear = nn.Linear() 
        
    def forward(self,sen_a,sent_b):
        sent_a_embedding = self.bert_model(sen_a)
        sent_b_embedding = self.bert_model(sent_b)
        pool_a = self.pooling(sent_a_embedding)
        pool_b = self.pooling(sent_b_embedding)
        concat_a_b = torch.cat((pool_a,pool_b,torch.abs(pool_a - pool_b)),dim=-1)
        linear = nn.Linear(concat_a_b.shape[-1],out_features=2)
        ln = linear(concat_a_b)
        res = F.softmax(ln)
        return res

