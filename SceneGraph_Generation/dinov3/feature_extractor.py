import torch.nn as nn
import torch

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        outputs = self.model(**inputs, output_hidden_states=True)
        
        patch_features = outputs.hidden_states[-1][:,5:,:]  # shape: (batch_size, num_patches, hidden_size)
        general_features = outputs.hidden_states[-1][:,0:5,:]
        #features = outputs.pooler_output
        
        return general_features, patch_features