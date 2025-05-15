import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertEnocder(nn.Module):
    def __init__(self):
        super(BertEnocder, self).__init__()
        
        self.bert = BertModel.from_pretrained("bert-large-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        
        self.max_length = 32
        
        for param in self.bert.parameters():
            param.requires_grad = False

    def tokenize(self, text):
        tokens = self.tokenizer(
            text, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def forward(self, text):
        input_ids, attention_mask = self.tokenize(text)
        input_ids, attention_mask = input_ids.to(next(self.parameters()).device), attention_mask.to(next(self.parameters()).device)

        with torch.no_grad():
            text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]


        return text_out