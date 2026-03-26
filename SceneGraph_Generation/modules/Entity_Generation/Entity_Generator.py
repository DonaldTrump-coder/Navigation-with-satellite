import torch.nn as nn
import torch
from SceneGraph_Generation.modules.Entity_Generation.Foundation_Encoders import CLIPEncoder
import torch.nn.functional as F

class Entity_Generator(nn.Module):
    def __init__(self, vector_dim, cnnmodel, vitmodel):
        super(Entity_Generator, self).__init__()
        self.vector_dim = vector_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.original_encoder = CLIPEncoder(vitmodel)
        self.entity_encoder = CLIPEncoder(cnnmodel)
        # cnn dim: 1024
        # vit dim: 512
        
        self.tau = 1
        
    def resize_2_n(self, x, n: int): # x: [batch, 2 * (vector_dim // 8 + 3 + 3 + 2) + 512]
        x = x.unsqueeze(1)
        x_resized = F.interpolate(
            x,
            size=n,
            mode='linear',
            align_corners=False
        )
        return x_resized.squeeze(1) # [batch, n]
        
    def forward(self,
                entity_features, # [batch, vector_dim // 8 + 3 + 3 + 2, entity_height, entity_width]
                entity_originals, # [batch, 3, entity_height, entity_width]
                originals # [batch, 3, height, width]
                ):
        entity_features = torch.cat([
                                    self.avg_pool(entity_features),
                                    self.max_pool(entity_features)
                                    ], dim=1)
        entity_features = entity_features.flatten(1) # [batch, 2 * (vector_dim // 8 + 3 + 3 + 2)]
        entity_originals = self.entity_encoder(entity_originals) # [batch, 1024]
        originals = self.original_encoder(originals) # [batch, 512]
        entity_features = torch.cat([entity_features, entity_originals], dim=1) # [batch, 2 * (vector_dim // 8 + 3 + 3 + 2) + 1024]
        unfused_entity_features = entity_features.clone() # [batch, 2 * (vector_dim // 8 + 3 + 3 + 2) + 1024]
        unfused_entity_features = F.normalize(unfused_entity_features, dim=-1)
        entity_features = self.resize_2_n(entity_features, 512) # [batch, 512]
        
        # features fusion from global and local in paper: ConceptFusion
        entity_features = F.normalize(entity_features, dim=-1) # [batch, 512]
        originals = F.normalize(originals, dim=-1) # [batch, 512]
        sim_matrix = entity_features @ entity_features.T
        batch_size = sim_matrix.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix[mask].view(batch_size, batch_size - 1)
        patch_sim = sim_matrix.mean(dim=-1) # [batch]
        global_sim = (originals * entity_features).sum(dim=-1) # [batch]
        w = (patch_sim + global_sim) / self.tau # [batch]
        weights = F.softmax(w, dim=0).unsqueeze(-1) # [batch, 1]
        entity_features = weights * originals + (1 - weights) * entity_features # [batch, 512]
        entity_features = torch.concat([entity_features, unfused_entity_features], dim=-1) # [batch, 2 * (vector_dim // 8 + 3 + 3 + 2) + 512]
        
        if self.training:
            # training mode
            pass
        else:
            # testing mode
            pass