import torch
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import v2
from torchvision import transforms

def resize_to_nearest_multiple_of_n(image, n):
    width, height = image.size
    
    new_width = int(round(width / n) * n)
    new_height = int(round(height / n) * n)
    #new_width = 256
    #new_height = 256
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def make_transform():
    resize_to_16 = lambda image: resize_to_nearest_multiple_of_n(image, 256)
    
    to_tensor = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    
    return v2.Compose([
        resize_to_16,
        to_tensor,
        to_float,
        normalize,
    ])

def model_loader(model_path):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, device_map="auto")
    return processor, model

def load_image_paths(image_folder):
    extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

class SatelliteDataset(Dataset):
    def __init__(self, image_paths, transform=None, patch_size=(128, 128)):
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 加载图像并转换为RGB格式
        
        if self.transform:
            image = self.transform(image)
        
        patch_width, patch_height = self.patch_size
        patches, indices = self._split_into_patches(image, patch_width, patch_height)
        inputs = {'pixel_values': patches, 'indices': indices, 'image_idx':idx, 'patches_num': patches.shape[0]}
        
        #inputs = {'pixel_values': image}
        #if inputs['pixel_values'].dim() == 4:
            #inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        
        return inputs
    
    def _split_into_patches(self, image, patch_width, patch_height):
        _, height, width = image.shape
        patches = []
        indices = []

        for i in range(0, height, patch_height):
            for j in range(0, width, patch_width):
                if i + patch_height <= height and j + patch_width <= width:
                    patch = image[:, i:i + patch_height, j:j + patch_width]

                    patches.append(patch)
                    indices.append(torch.tensor([i, j], dtype=torch.int64))
        
        patches = torch.stack(patches, dim=0)
        indices = torch.stack(indices, dim=0)
        
        return patches, indices

if __name__ == "__main__":
    model_path = "../models/dinov3_vitl16_pretrain_sat493m"
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, device_map="auto")
    print(model)