from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import cv2
import torch
import torch.nn.functional as F

"""
"shapes": 
    [
        {
            "label": ,
            "points": [
                [],
                []...
            ],
            "group_id": ,
            "description": ,
            "shape_type": ,
            "flags": {},
            "mask": ,
            "dx": ,
            "dy": ,
        },
       ...
    ]
"""
class Patch_features_dataset(Dataset):
    def __init__(self,
                 fused_entity_features,
                 shapes,
                 text_processor,
                 max_length = 8
                 ):
        self.text_processor = text_processor
        self.max_length = max_length
        
        self.labels = []
        self.dxs = []
        self.dys = []
        self.fused_entity_features = fused_entity_features
        
        self.patch_num = len(shapes)
        for shape in shapes:
            self.labels.append(shape["label"])
            self.dxs.append(shape["dx"])
            self.dys.append(shape["dy"])
        
    def __len__(self):
        return self.patch_num
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        dx = self.dxs[idx]
        dy = self.dys[idx]
        encoding = self.text_processor.tokenizer(
            label,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoding.input_ids.squeeze(0) # [L]
        attention_mask = encoding.attention_mask.squeeze(0) # [L]
        labels = input_ids.clone() # for loss computation
        pad_token_id = self.text_processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        offset = torch.tensor([dx, dy], dtype=torch.float32)
        fused_entity_feature = self.fused_entity_features[idx]
        return {
            "fused_entity_features": fused_entity_feature,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "offset": offset # [2]
        }
class Patches_dataset(Dataset):
    def __init__(self,
                 npy_path,
                 tif_path,
                 json_path,
                 cnn_preprocess,
                 vit_preprocess,
                 roi_size = (256, 256)
                 ):
        self.npy_data = np.load(npy_path)
        _, h, w = self.npy_data.shape
        pos_embed = get_2d_sincos_pos_embed(h, w, embed_dim=64)
        self.npy_data = np.concatenate([self.npy_data, pos_embed], axis=0) # positional embeddings
        
        self.vit_preprocess = vit_preprocess
        self.cnn_preprocess = cnn_preprocess
        
        self.image = Image.open(tif_path)
        original_width, original_height = self.image.size
        self.image = self.image.resize(
            (w, h),
            resample=Image.LANCZOS
        )
        resize_w = w / original_width
        resize_h = h / original_height # from original img to resized img
        
        self.entity_features = []
        self.entity_originals = []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)
        shapes = self.json_data["shapes"]
        self.patch_num = len(shapes)
        for shape in shapes:
            points = np.array(shape["points"], dtype=np.float32) # [N, 2]
            points[:, 0] *= resize_w # x
            points[:, 1] *= resize_h # y
            shape["dx"] = shape["dx"] * resize_w
            shape["dy"] = shape["dy"] * resize_h
            x_min = int(np.min(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            x_max = int(np.max(points[:, 0]))
            y_max = int(np.max(points[:, 1]))
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            shape["dx"] = shape["dx"] / (x_max - x_min)
            shape["dy"] = shape["dy"] / (y_max - y_min)
            # cropping for img and feature map
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            image_np = np.array(self.image)
            image_np = image_np * mask[:, :, None]
            image_np = image_np[y_min:y_max, x_min:x_max]
            features_np = self.npy_data.copy()
            features_np = features_np * mask[None, :, :]
            features_np = features_np[:, y_min:y_max, x_min:x_max]
            
            image_patch = Image.fromarray(image_np)
            image_patch = resize_img_with_padding(image_patch, roi_size)
            image_patch.show()
            feature_patch = torch.tensor(features_np, dtype=torch.float32)
            feature_patch = resize_feature_with_padding(feature_patch, roi_size)
            
            self.entity_features.append(feature_patch)
            self.entity_originals.append(image_patch)
        self.shapes = shapes
        
    def __len__(self):
        return self.patch_num
    
    def get_shapes(self):
        return self.shapes
    
    def __getitem__(self, idx):
        entity_feature = self.entity_features[idx]
        entity_original = self.entity_originals[idx]
        image = self.image
        image = self.vit_preprocess(image)
        entity_original = self.cnn_preprocess(entity_original)
        return {
            "image": image, # preprocessed original img
            "entity_feature": entity_feature,
            "entity_original": entity_original
        }
    
class Patches_relation_dataset(Dataset): # load data for relation prediction
    def __init__(self,
                 json_path,
                 fused_entity_features,
                 shapes
                 ):
        self.fused_entity_features = fused_entity_features
        self.pairs = []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)
        self.patch_num = len(shapes)
        connections = self.json_data["connections"]
        self.adj_matrix = torch.zeros((self.patch_num, self.patch_num), dtype=torch.float32)
        for i, j in connections:
            self.adj_matrix[i-1, j-1] = 1
            self.adj_matrix[j-1, i-1] = 1
        for i in range(self.patch_num):
            for j in range(self.patch_num):
                if i == j:
                    continue
                self.pairs.append((i, j)) # load all pairs
            
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        fused_entity_features_i = self.fused_entity_features[i]
        fused_entity_features_j = self.fused_entity_features[j]
        label = self.adj_matrix[i, j]
        return {
            "fused_entity_features": torch.stack([fused_entity_features_i, fused_entity_features_j], dim=0),
            "label": label
        }
    
def get_2d_sincos_pos_embed(h, w, embed_dim=64):
    assert embed_dim % 4 == 0, f"embed_dim should be divisible by 4, got {embed_dim}"
    
    dim = embed_dim // 2
    y = np.arange(h)
    x = np.arange(w)
    grid_y, grid_x = np.meshgrid(y, x, indexing='ij')  # (H, W)
    grid_x = grid_x / w
    grid_y = grid_y / h
    
    omega = np.arange(dim // 2)
    omega = 1. / (10000 ** (omega / (dim // 2)))
    pos_x = grid_x[..., None] * omega  # (H, W, dim/2)
    pos_y = grid_y[..., None] * omega
    
    pos_x = np.concatenate([np.sin(pos_x), np.cos(pos_x)], axis=-1)
    pos_y = np.concatenate([np.sin(pos_y), np.cos(pos_y)], axis=-1)
    pos = np.concatenate([pos_x, pos_y], axis=-1)
    return pos.transpose(2, 0, 1)  # (dim, H, W)

def resize_img_with_padding(img, target_size):
    target_w, target_h = target_size
    w, h = img.size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img

def resize_feature_with_padding(feature, target_size):
    C, H, W = feature.shape
    target_h, target_w = target_size
    
    scale = min(target_h / H, target_w / W)
    new_h = int(H * scale)
    new_w = int(W * scale)
    
    feature = feature.unsqueeze(0)  # [1, C, H, W]
    resized = F.interpolate(
        feature,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    padded = torch.zeros((C, target_h, target_w), dtype=resized.dtype)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    padded[:, y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded