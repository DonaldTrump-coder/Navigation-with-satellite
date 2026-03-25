import torch.nn as nn
from SceneGraph_Generation.dinov3.feature_extractor import FeatureExtractor
from SceneGraph_Generation.dinov3.loader import model_loader
from SceneGraph_Generation.modules.Expander import Expander, resizer, resize_origin, splitter
import torch
import torch.nn.functional as F
#from visualizer.features_visualizer import FeaturesVisualizer

class EntityDetector(nn.Module):
    def __init__(self, dino_path, vector_dim):
        super(EntityDetector, self).__init__()
        self.conv_extractor = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.dino_path = dino_path
        self.vector_dim = vector_dim
        _, self.model = model_loader(self.dino_path)
        self.dino_module = FeatureExtractor(self.model)
        self.expander = None
        
        self.upconv1 = nn.ConvTranspose2d(2 * self.vector_dim, self.vector_dim, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(self.vector_dim, self.vector_dim // 2, kernel_size=2, stride=2)
        self.connect_conv1 = nn.Sequential(
            nn.Conv2d(2 * self.vector_dim, self.vector_dim // 2, kernel_size=1),
            nn.BatchNorm2d(self.vector_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.ConvTranspose2d(self.vector_dim // 2, self.vector_dim // 4, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(self.vector_dim // 4, self.vector_dim // 8, kernel_size=2, stride=2)
        self.connect_conv2 = nn.Sequential(
            nn.Conv2d(self.vector_dim // 2, self.vector_dim // 8, kernel_size=1),
            nn.BatchNorm2d(self.vector_dim // 8),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(self.vector_dim // 8 + 3 + 3, 1, kernel_size=1)
        )
        
    def get_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        self.device = self.get_device()
        
        x = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in x.items()}
        batch_size, num_patches, channels, patch_height, patch_width = x['pixel_values'].shape
        x['pixel_values'] = resize_origin(x['pixel_values'], x['indices'])
        # restored x
        original_image = x['pixel_values']
        x['pixel_values'] = self.conv_extractor(original_image) # [batch_size, 3, height, width]
        conved_image = x['pixel_values']
        
        x['pixel_values'] = splitter(x['pixel_values'], x['indices'], patch_height, patch_width) # [batch_size x num_of_patches x channels x patch_height x patch_width]
        x['pixel_values'] = x['pixel_values'].view(batch_size * num_patches, channels, patch_height, patch_width)

        general_features, patch_features = self.dino_module(x) # [num_of_patches x num_of_vectors x length_of_vector]
        
        if self.expander is None:
            self.expander = Expander(patch_num_vectors=patch_features.shape[-2],
                                num_patches=num_patches,
                                vector_dim=general_features.shape[-1],
                                num_heads=4,
                                dropout=0.1
                                ).to(self.device)
        general_features = self.expander(general_features) # [batch_size, num_patches, patch_num_vectors, vector_dim]

        patch_features = patch_features.view(batch_size, num_patches, patch_features.shape[-2], patch_features.shape[-1])

        num_of_vec_in_patch_height = int(patch_height / 16)
        num_of_vec_in_patch_width = int(patch_width / 16)
        general_features = general_features.view(batch_size, num_patches, num_of_vec_in_patch_height, num_of_vec_in_patch_width, -1)
        patch_features = patch_features.view(batch_size, num_patches, num_of_vec_in_patch_height, num_of_vec_in_patch_width, -1) # [batch_size x num_of_patches x num_of_vectors_in_patch_height x num_of_vectors_in_patch_width x length_of_vector]
        # reshape the patch_features in each patch from 1D to 2D
        patch_features = torch.cat((patch_features, general_features), dim=-1)
        
        features = resizer(patch_features, x['indices']) # [batch_size, height, width, vector_dim]    vectoe_dim = 2048
        features = features.permute(0, 3, 1, 2) # [batch_size, vector_dim, height, width]
        
        # U-Net like decoder
        # UpSampling
        original_features = features
        features = F.relu(self.upconv1(features))
        features = F.relu(self.upconv2(features))
        original_features = F.interpolate(original_features, size = features.shape[2:], mode='bilinear', align_corners=False)
        original_features = self.connect_conv1(original_features)
        features = features + original_features # residual connection [batch_size, vector_dim // 2, height, width]
        
        original_features = features
        features = F.relu(self.upconv3(features))
        features = F.relu(self.upconv4(features))
        original_features = F.interpolate(original_features, size = features.shape[2:], mode='bilinear', align_corners=False)
        original_features = self.connect_conv2(original_features)
        features = features + original_features # residual connection [batch_size, vector_dim // 8, height, width]
        
        features = torch.concatenate([features, conved_image, original_image], dim=1) # [batch_size, vector_dim // 8 + 3 + 3, height, width] features map
        
        logits = self.classifier(features) # [batch_size, 1, height, width]
        
        return logits, features