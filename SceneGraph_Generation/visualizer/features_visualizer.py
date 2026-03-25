import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class FeaturesVisualizer:
    def __init__(self, features):
        self.features = features

    def visualize(self): # Feature Map
        pooled_feature = self.features.mean(dim=-1)
        plt.imshow(pooled_feature[0, :, :].cpu().detach().numpy(), cmap='hot')
        plt.colorbar()
        plt.title("Average Pooled Feature Map")
        plt.show()
        
    def visualize_img(self): # RGB
        img = self.features[0].permute(1, 2, 0).cpu().detach().numpy()
        
        if img.max() <= 1.0:
            img = img * 255
        
        img = Image.fromarray(img.astype(np.uint8))
        img.show()