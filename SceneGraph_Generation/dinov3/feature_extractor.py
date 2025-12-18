import torch.nn as nn
import torch

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        features = outputs.hidden_states[-1][:,5:,:]  # shape: (batch_size, num_patches, hidden_size)
        #features = outputs.pooler_output
        
        return features