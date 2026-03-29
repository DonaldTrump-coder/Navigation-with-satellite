import torch
import numpy as np
from SceneGraph_Generation.Scene_graph_generator import EntityDetector

def navigator(img: np.ndarray):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    dino_path = "./SceneGraph_Generation/models/dinov3_vitl16_pretrain_sat493m"
    dino_dim = 1024
    
    detector_path = "./SceneGraph_Generation/models/Entity_Generator/model.pt"
    detector = EntityDetector(dino_path, dino_dim).to(device=device)
    detector_state_dict = torch.load(
        detector_path,
        map_location="cpu"
    )
    detector.load_state_dict(detector_state_dict)
    detector.eval()
    detector.to(device)
    
if __name__ == "__main__":
    img = None
    navigator(img=img)