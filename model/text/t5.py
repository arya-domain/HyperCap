import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel

class T5Encoder(nn.Module):
    def __init__(self):
        super(T5Encoder, self).__init__()
        
        self.model = T5EncoderModel.from_pretrained("google-t5/t5-large")
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
        
        self.max_length = 32
        
        for param in self.model.parameters():
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
            text_out = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]


        return text_out