import torch.nn as nn
import torch

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for _, param in self.model.named_parameters():
            param.requires_grad = False
            
        for i in [22,23]:
            for param in self.model.model.layer[i].parameters():
                param.requires_grad = True
                
        for param in self.model.norm.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True)
        
        patch_features = outputs.hidden_states[-1][:,5:,:]  # shape: (batch_size, num_patches, hidden_size)
        general_features = outputs.hidden_states[-1][:,0:5,:]
        #features = outputs.pooler_output
        
        return general_features, patch_features