from dinov3.loader import model_loader, load_image_paths, make_transform, SatelliteDataset
from torch.utils.data import DataLoader
import torch
from Scene_graph_generator import EntityDetector
from visualizer.features_visualizer import FeaturesVisualizer

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_path = "SceneGraph_Generation/models/dinov3_vitl16_pretrain_sat493m"
    image_folder = "SceneGraph_Generation/data"
    label_folder = ""

    image_paths = load_image_paths(image_folder)
    transform = make_transform()
    dataset = SatelliteDataset(image_paths, label_folder, transform, patch_size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    dino_dim = 1024

    model = EntityDetector(model_path, dino_dim).to(device=device)

    for batch in dataloader:
        model(batch)
        break