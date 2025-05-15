from ..model.vision import FAHM
import torch.nn as nn

class VisionModel(nn.Module):
    def __init__(self, patch_size=11, bands=200, output_dimension=768, return_type_list=True):
        super().__init__()
        self.return_type_list = return_type_list
        self.base_vision = FAHM.FAHM(img_size=patch_size, in_chans=bands, n_groups=[2, 2, 2], depths=[1, 1, 1])
        self.fc = nn.Linear(64, output_dimension)

    def forward(self, pixel_values,
            output_attentions = None,
            output_hidden_states= None,
            return_dict= None,
            interpolate_pos_encoding= None,):
        x = self.base_vision(pixel_values)
        x = self.fc(x)

        if self.return_type_list:
            return [x.unsqueeze(1)]
        else:
            return x.unsqueeze(1)