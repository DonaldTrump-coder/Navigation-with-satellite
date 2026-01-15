from dinov3.loader import model_loader, load_image_paths, make_transform, SatelliteDataset
from dinov3.feature_extractor import FeatureExtractor
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    model_path = "models/dinov3_vitl16_pretrain_sat493m"
    image_folder = "data/examples"
    processor, model = model_loader(model_path)
    image_paths = load_image_paths(image_folder)
    dino_module = FeatureExtractor(model)
    transform = make_transform()
    dataset = SatelliteDataset(image_paths, transform, patch_size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        batch_size, num_patches, channels, patch_height, patch_width = batch['pixel_values'].shape
        batch['pixel_values'] = batch['pixel_values'].view(batch_size * num_patches, channels, patch_height, patch_width)
        outputs = dino_module(batch)