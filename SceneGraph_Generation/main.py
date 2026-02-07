from dinov3.loader import model_loader, load_image_paths, make_transform, SatelliteDataset
from dinov3.feature_extractor import FeatureExtractor
from torch.utils.data import DataLoader
import torch
from modules.Expander import Expander, resizer

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_path = "SceneGraph_Generation/models/dinov3_vitl16_pretrain_sat493m"
    image_folder = "SceneGraph_Generation/data"
    processor, model = model_loader(model_path)
    image_paths = load_image_paths(image_folder)
    dino_module = FeatureExtractor(model)
    transform = make_transform()
    dataset = SatelliteDataset(image_paths, transform, patch_size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        batch_size, num_patches, channels, patch_height, patch_width = batch['pixel_values'].shape
        batch['pixel_values'] = batch['pixel_values'].view(batch_size * num_patches, channels, patch_height, patch_width)

        general_features, patch_features = dino_module(batch) # [num_of_patches x num_of_vectors x length_of_vector]
        expander = Expander(patch_num_vectors=patch_features.shape[-2],
                            num_patches=num_patches,
                            vector_dim=general_features.shape[-1],
                            num_heads=4,
                            dropout=0.1
                            ).to(device)
        general_features = expander(general_features) # [batch_size, num_patches, patch_num_vectors, vector_dim]

        patch_features = patch_features.view(batch_size, num_patches, patch_features.shape[-2], patch_features.shape[-1])

        num_of_vec_in_patch_height = int(patch_height / 16)
        num_of_vec_in_patch_width = int(patch_width / 16)
        general_features = general_features.view(batch_size, num_patches, num_of_vec_in_patch_height, num_of_vec_in_patch_width, -1)
        patch_features = patch_features.view(batch_size, num_patches, num_of_vec_in_patch_height, num_of_vec_in_patch_width, -1) # [batch_size x num_of_patches x num_of_vectors_in_patch_height x num_of_vectors_in_patch_width x length_of_vector]
        # reshape the patch_features in each patch from 1D to 2D
        patch_features = torch.cat((patch_features, general_features), dim=-1)
        
        features = resizer(patch_features, batch['indices']) # [batch_size, height, width, vector_dim]