import torch.nn.functional as F
import torch.nn as nn
import torch

class Expander(nn.Module):
    def __init__(self, patch_num_vectors, num_patches, vector_dim, num_heads = 4, dropout=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.vector_dim = vector_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = nn.MultiheadAttention(vector_dim, num_heads, dropout=dropout)
        self.query_transform = nn.Linear(vector_dim, vector_dim)
        self.key_transform = nn.Linear(vector_dim, vector_dim)
        self.value_transform = nn.Linear(vector_dim, vector_dim)
        self.patch_num_vectors = patch_num_vectors

    def forward(self, x):
        _, _, vector_dim = x.shape
        x = x.permute(0, 2, 1)
        x = (F.adaptive_avg_pool1d(x, 1)).view(-1, self.num_patches, 1, vector_dim) # [batch_size, num_patches, 1, vector_dim]
        x = x.squeeze(2) # [batch_size, num_patches, vector_dim]
        query = self.query_transform(x).permute(1, 0, 2)
        key = self.key_transform(x).permute(1, 0, 2)
        value = self.value_transform(x).permute(1, 0, 2)
        x, _ = self.attention(query, key, value)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(2) # [batch_size, num_patches, 1, vector_dim]
        x = x.repeat(1, 1, self.patch_num_vectors, 1) # [batch_size, num_patches, patch_num_vectors, vector_dim]
        return x
    

def resizer(features, # [batch_size x num_of_patches x num_of_vectors_in_patch_height x num_of_vectors_in_patch_width x length_of_vector]
            indices # [batch_size x num_of_patches x 2]
            ):
    batch_size, num_of_patches, num_vectors_height, num_vectors_width, vector_dim = features.shape

    max_y = int(indices[:, :, 0].max().item() / 16) # max y index
    max_x = int(indices[:, :, 1].max().item() / 16) # max x index

    height = max_y + num_vectors_height
    width = max_x + num_vectors_width

    # create a new tensor with zeros
    new_features = torch.zeros(batch_size, height, width, vector_dim).to(features.device)
    y_coords, x_coords = indices[:, :, 0], indices[:, :, 1]  # [batch_size x num_of_patches] [batch_size x num_of_patches]

    y_coords = y_coords.unsqueeze(2).unsqueeze(3)
    x_coords = x_coords.unsqueeze(2).unsqueeze(3) # [batch_size, num_of_patches, 1, 1]

    for i in range(num_of_patches):
        y = (y_coords[:, i]/16).to(torch.int)
        x = (x_coords[:, i]/16).to(torch.int) # [batch_size, 1, 1]
        patch = features[:, i]
        new_features[:, y:y+num_vectors_height, x:x+num_vectors_width, :] = patch

    return new_features # [batch_size, height, width, vector_dim]

def resize_origin(features, # [batch_size, num_patches, channels, patch_height, patch_width]
                  indices # [batch_size, num_of_patches, 2]
                  ):
    batch_size, num_patches, channels, patch_height, patch_width = features.shape
    max_y = int(indices[:, :, 0].max().item()) # max y index
    max_x = int(indices[:, :, 1].max().item()) # max x index
    
    height = max_y + patch_height
    width = max_x + patch_width
    
    new_features = torch.zeros(batch_size, channels, height, width).to(features.device)
    
    for b in range(batch_size):
        for i in range(num_patches):
            y = int(indices[b, i, 0])
            x = int(indices[b, i, 1])
            
            patch = features[b, i]
            new_features[b, :, y:y+patch_height, x:x+patch_width] = patch
    
    return new_features # [batch_size, channels, height, width]

def splitter(features, # [batch_size, channels, height, width]
             indices, # [batch_size x num_of_patches x 2]
             patch_height, # int
             patch_width, # int
             ):
    batch_size, channels, height, width = features.shape
    batch_size, num_of_patches, _ = indices.shape
    new_features = torch.zeros(batch_size, num_of_patches, channels, patch_height, patch_width).to(features.device)
    
    y_coords, x_coords = indices[:, :, 0], indices[:, :, 1]  # [batch_size x num_of_patches] [batch_size x num_of_patches]
    
    for i in range(num_of_patches):
        y = (y_coords[:, i]).to(torch.int)
        x = (x_coords[:, i]).to(torch.int) # [batch_size, 1, 1]
        
        patch = features[:, :, y:y+patch_height, x:x+patch_width] # [batch_size, channels, patch_height, patch_width]
        new_features[:, i] = patch
    
    return new_features # [batch_size, num_of_patches, channels, patch_height, patch_width]