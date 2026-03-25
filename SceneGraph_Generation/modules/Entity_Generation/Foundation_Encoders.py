import torch.nn as nn
import torch

class CLIPEncoder(nn.Module):
    def __init__(self, clipmodel, ckpt_file):
        super(CLIPEncoder, self).__init__()
        self.model = clipmodel
        self.model.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu')))
        
    def forward(self, image):
        image_features = self.model.encode_image(image)
        return image_features