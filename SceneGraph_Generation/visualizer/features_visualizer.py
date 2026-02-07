import torch
import matplotlib.pyplot as plt

class FeaturesVisualizer:
    def __init__(self, features):
        self.features = features

    def visualize(self):
        pooled_feature = self.features.mean(dim=-1)
        plt.imshow(pooled_feature[:, :].cpu().detach().numpy(), cmap='hot')
        plt.colorbar()
        plt.title("Average Pooled Feature Map")
        plt.show()