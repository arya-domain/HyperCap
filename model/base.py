import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention

class Model(nn.Module):
    def __init__(self, num_classes, vision_encoder, text_encoder, vision_out, text_out, merging_method, use_only=None):
        super(Model, self).__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.text_fc = nn.Linear(text_out, vision_out)
        # self.vision_fc = nn.Linear(vision_out, vision_out)

        self.merging_method = merging_method
        self.use_only = use_only  # Choose modality: "vision", "text", or None (both)

        # Fusion Method
        if merging_method == 'MHA':  # Multihead Attention
            self.cross_attention = nn.MultiheadAttention(embed_dim=vision_out, num_heads=8, batch_first=True)
            input_dim = vision_out
        elif merging_method == 'CA':  # Cross Attention
            self.cross_attention = Attention(query_dim=1, heads=8, dim_head=64, dropout=0.0,
                                             bias=False, cross_attention_dim=vision_out, upcast_attention=True,
                                             out_bias=True)
            input_dim = vision_out
        elif merging_method == 'CONCAT':  # Concatenation
            input_dim = vision_out * 2
        else:  # PWA (Point-wise Addition) and PWM (Point-wise Multiplication)
            input_dim = vision_out

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, image, text):
        # Process single modality if specified
        if self.use_only == "vision":
            return self.fc(self.vision_encoder(image).view(image.shape[0], -1))
            # return self.fc(self.vision_fc(self.vision_encoder(image).view(image.shape[0], -1)))
        if self.use_only == "text":
            return self.fc(self.text_fc(self.text_encoder(text)))

        # Process both modalities
        text_emb = self.text_fc(self.text_encoder(text))
        vision_emb = self.vision_encoder(image).view(image.shape[0], -1)
        # vision_emb = self.vision_fc(self.vision_encoder(image).view(image.shape[0], -1))

        # Merge features based on the selected method
        merge_fn = getattr(self, f"merge_{self.merging_method.lower()}", None)
        if merge_fn:
            cls_output = merge_fn(text_emb, vision_emb)
        else:
            raise ValueError(f"Unknown merging method: {self.merging_method}")

        return self.fc(cls_output)

    def merge_mha(self, text_emb, vision_emb):
        cls_output, _ = self.cross_attention(text_emb, vision_emb, vision_emb)
        return cls_output + vision_emb

    def merge_ca(self, text_emb, vision_emb):
        vision_emb = vision_emb.unsqueeze(1).unsqueeze(1)
        text_emb = text_emb.unsqueeze(1)
        cls_output = self.cross_attention(vision_emb, text_emb)
        return cls_output.squeeze(1).squeeze(1) + vision_emb.squeeze(1).squeeze(1)

    def merge_concat(self, text_emb, vision_emb):
        return torch.cat((vision_emb, text_emb), dim=1)

    def merge_pwa(self, text_emb, vision_emb):
        return text_emb + vision_emb

    def merge_pwm(self, text_emb, vision_emb):
        return text_emb * vision_emb
