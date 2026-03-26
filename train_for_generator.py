import torch
import os
import random

# json data:
"""
{
    "version": ,
    "flags": {},
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
    ],
    "imagePath": ,
    "imageData": ,
    "imageHeight": ,
    "imageWidth": ,
    "connections": [[1, 2], [1, 3], ...],
    "traj": [[[x1, y1], [x2, y2], ...], [], ...]
}
"""


def main():
    train_ratio = 0.8
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    cnn_ckpt = "./SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-RN50.pt"
    vit_ckpt = "./SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-ViT-B-32.pt"
    features_folder = "./outputs"
    data_folder = "./data/Google"
    
    feature_paths = []
    tif_paths = []
    json_paths = []
    for root, _, files in os.walk(features_folder):
        for f in files:
            if f.endswith(".npy"):
                npy_path = os.path.join(root, f)
                rel_path = os.path.relpath(npy_path, features_folder)
                tif_rel_path = os.path.splitext(rel_path)[0] + ".tif"
                tif_path = os.path.join(data_folder, tif_rel_path)
                tif_dir = os.path.dirname(tif_path)
                json_name = os.path.splitext(os.path.basename(tif_path))[0] + ".json"
                json_path = os.path.join(tif_dir, "labels", json_name)
                
                feature_paths.append(npy_path)
                tif_paths.append(tif_path)
                json_paths.append(json_path)
    
    samples = list(zip(feature_paths, tif_paths, json_paths))
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    for scene in train_samples:
        npy_path, tif_path, json_path = scene
    
if __name__ == '__main__':
    main()