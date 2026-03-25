import torch.nn as nn

class Entity_Generator(nn.Module):
    def __init__(self):
        super(Entity_Generator, self).__init__()
        
    def forward(self,
                entity_features, # [batch, vector_dim // 8 + 3 + 3, entity_height, entity_width]
                entity_originals, # [batch, 3, entity_height, entity_width]
                originals # [batch, 3, height, width]
                ):
        pass