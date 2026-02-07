import torch.nn as nn
from dinov3.feature_extractor import FeatureExtractor
from dinov3.loader import model_loader
from modules.Expander import Expander, resizer
import torch

class Scene_graph_generator(nn.Module):
    def __init__(self, dino_path):
        super(Scene_graph_generator, self).__init__()
        self.dino_path = dino_path
        _, self.model = model_loader(self.dino_path)
        self.dino_module = FeatureExtractor(self.model)
        self.device = self.get_device()

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        batch_size, num_patches, channels, patch_height, patch_width = x['pixel_values'].shape
        x['pixel_values'] = x['pixel_values'].view(batch_size * num_patches, channels, patch_height, patch_width)

        general_features, patch_features = self.dino_module(x) # [num_of_patches x num_of_vectors x length_of_vector]
        
        expander = Expander(patch_num_vectors=patch_features.shape[-2],
                            num_patches=num_patches,
                            vector_dim=general_features.shape[-1],
                            num_heads=4,
                            dropout=0.1
                            ).to(self.device)
        general_features = expander(general_features) # [batch_size, num_patches, patch_num_vectors, vector_dim]

        patch_features = patch_features.view(batch_size, num_patches, patch_features.shape[-2], patch_features.shape[-1])

        num_of_vec_in_patch_height = int(patch_height / 16)
        num_of_vec_in_patch_width = int(patch_width / 16)
        general_features = general_features.view(batch_size, num_patches, num_of_vec_in_patch_height, num_of_vec_in_patch_width, -1)
        patch_features = patch_features.view(batch_size, num_patches, num_of_vec_in_patch_height, num_of_vec_in_patch_width, -1) # [batch_size x num_of_patches x num_of_vectors_in_patch_height x num_of_vectors_in_patch_width x length_of_vector]
        # reshape the patch_features in each patch from 1D to 2D
        patch_features = torch.cat((patch_features, general_features), dim=-1)
        
        features = resizer(patch_features, x['indices']) # [batch_size, height, width, vector_dim]