import torch, open_clip
import numpy as np
from SceneGraph_Generation.Scene_graph_generator import EntityDetector
from SceneGraph_Generation.dinov3.loader import make_transform, SatelliteDataset_infer
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from SceneGraph_Generation.modules.Entity_splitter import split_entities
from SceneGraph_Generation.modules.Languagemodels.GLMOCR import load_ocr_model
from transformers import AutoProcessor
from SceneGraph_Generation.modules.Entity_Generation.Entity_Generator import Entity_Generator, Feature_Fuser
from SceneGraph_Generation.datasets import Patches_dataset_infer, Patch_features_dataset_infer
import cv2
import os
from scipy.sparse import csr_matrix
from SceneGraph_Generation.Scene_graph import SceneGraph, pix2geo
from networkx import DiGraph
from networkx.algorithms.tree import minimum_spanning_arborescence

def navigator(img: np.ndarray, min_lon, max_lon, min_lat, max_lat): # [H, W, C]
    img = Image.fromarray(img)
    
    # generate the Scene Graph
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    dino_path = "./SceneGraph_Generation/models/dinov3_vitl16_pretrain_sat493m"
    dino_dim = 1024
    max_length = 8
    transform = make_transform()
    
    detector_path = "./SceneGraph_Generation/models/Entity_Detector/model.pt"
    detector = EntityDetector(dino_path, dino_dim).to(device=device)
    detector_state_dict = torch.load(
        detector_path,
        map_location="cpu"
    )
    expander_state_dict = {k.replace("expander.", ""): v 
                       for k, v in detector_state_dict.items() if k.startswith("expander.")}
    detector.expander_state_dict = expander_state_dict
    filtered_state_dict = {k: v for k, v in detector_state_dict.items() if not k.startswith("expander.")}
    detector.load_state_dict(filtered_state_dict, strict=False)
    detector.eval()
    detector.to(device)
    
    dataset = SatelliteDataset_infer(img, transform, (256, 256))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        logits, features = detector(batch)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    preds = preds[0]
    mask = preds.squeeze(0).cpu().numpy()
    features = features[0].detach().cpu().numpy()
    filtered_masks = split_entities(mask) # [mask1, mask2 ...]
    
    node_num = len(filtered_masks)
    
    roi_size = (256, 256)
    cnn_ckpt = "./SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-RN50.pt"
    vit_ckpt = "./SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-ViT-B-32.pt"
    cnn_model, _, cnn_preprocess = open_clip.create_model_and_transforms('RN50', pretrained=cnn_ckpt)
    vit_model, _, vit_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=vit_ckpt)
    text_decoder_path = "./SceneGraph_Generation/models/GLM-OCR"
    text_decoder = load_ocr_model(text_decoder_path)
    text_processor = AutoProcessor.from_pretrained(text_decoder_path)
    
    generator_path = "./SceneGraph_Generation/models/Entity_Generator/model.pt"
    generator_state_dict = torch.load(
        generator_path,
        map_location="cpu"
    )
    
    model = Feature_Fuser(vector_dim=dino_dim, cnnmodel=cnn_model, vitmodel=vit_model).to(device)
    dataset = Patches_dataset_infer(features=features,
                                    img=img,
                                    masks=filtered_masks,
                                    cnn_preprocess=cnn_preprocess,
                                    vit_preprocess=vit_preprocess,
                                    roi_size=roi_size,
                                    )
    patch_batch_size = dataset.__len__()
    patch_dataloader = DataLoader(dataset, batch_size=patch_batch_size)
    for batch in patch_dataloader:
        image = batch["image"].to(device)
        entity_feature = batch["entity_feature"].to(device)
        entity_original = batch["entity_original"].to(device)
        model.eval()
        with torch.no_grad():
            fused_entity_features = model(entity_feature,
                                          entity_original,
                                          image)
    
    model = Entity_Generator(vector_dim=dino_dim,
                             ocrmodel=text_decoder,
                             lora_config=None,
                             max_length=max_length
                             ).to(device)
    model.load_state_dict(generator_state_dict)
    model.eval()
    model.inferring = True
    dataset = Patch_features_dataset_infer(fused_entity_features)
    dataloader = DataLoader(dataset, batch_size=patch_batch_size)
    for batch in dataloader:
        fused_entity_feature = batch.to(device)
        offsets, generated_ids = model(fused_entity_feature)
    offsets = offsets.detach().cpu().numpy()
    generated_ids = generated_ids.detach().cpu()
    texts = text_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True) # texts list
    texts = [text.replace("\n", "").replace("_", "") for text in texts]
    
    x_center_original = []
    y_center_original = []
    descriptions = [] # list[str]
    original_masks = []
    original_bounds = []
    patches_origin = []
    lons = []
    lats = []
    dist_matrix = np.zeros((node_num, node_num), dtype=np.float32)
    original_width, original_height = img.size
    h, w = mask.shape
    h_scale, w_scale = original_height / h, original_width / w
    patch_height = 256
    save_folder = "./outputs/test"
    os.makedirs(save_folder, exist_ok=True)
    
    for idx, filtered_mask in enumerate(filtered_masks):
        # the idx-th patch
        patch_folder = os.path.join(save_folder, f"{idx:03d}")  # 000, 001, 002 ...
        os.makedirs(patch_folder, exist_ok=True)
        
        ys, xs = np.nonzero(filtered_mask)
        x_min = int(xs.min())
        y_min = int(ys.min())
        x_max = int(xs.max())
        y_max = int(ys.max())
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        dx, dy = offsets[idx][0], offsets[idx][1]
        dx = dx * (x_max - x_min)
        dy = dy * (y_max - y_min)
        x_center += dx
        y_center += dy
        x_center *= w_scale
        y_center *= h_scale
        x_center_original.append(x_center)
        y_center_original.append(y_center)
        
        transformed_mask = filtered_mask.astype(np.uint8)
        transformed_mask = cv2.resize(transformed_mask,
                                      (original_width, original_height),
                                      interpolation=cv2.INTER_NEAREST
                                      ) # [original_height, original_width]
        original_masks.append(transformed_mask)
        
        contours, _ = cv2.findContours(
            transformed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        boundary = np.vstack([c[:, 0, :] for c in contours])  # [points_num, 2]
        original_bounds.append(boundary)
        
        x_min_patch = int(np.floor(x_min * w_scale))
        x_max_patch = int(np.ceil(x_max * w_scale))
        y_min_patch = int(np.floor(y_min * h_scale))
        y_max_patch = int(np.ceil(y_max * h_scale))
        x_min_patch = max(0, x_min_patch)
        y_min_patch = max(0, y_min_patch)
        x_max_patch = min(original_width, x_max_patch)
        y_max_patch = min(original_height, y_max_patch)
        patch = img.crop((x_min_patch, y_min_patch, x_max_patch, y_max_patch))
        w, h = patch.size
        patch_width = int(w * (patch_height / h))
        patch = patch.resize((patch_width, patch_height), Image.BILINEAR)
        patches_origin.append(patch)
        
        lon, lat = pix2geo(x_center, y_center, min_lon, max_lon, min_lat, max_lat, original_width, original_height)
        lons.append(lon)
        lats.append(lat)
        
        patch_path = os.path.join(patch_folder, f"patch.png")
        patch.save(patch_path)
    
    mask_path = os.path.join(save_folder, "mask.png")
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(mask_path)
        
    # constructing relation (distance) matrix
    for i in range(node_num):
        center = np.array([x_center_original[i], y_center_original[i]])
        for j in range(node_num):
            if i == j:
                continue
            boundary = original_bounds[j]
            distances = np.linalg.norm(boundary - center, axis=1)
            dist_matrix[i, j] = distances.min()
        
    # constructing connection
    np.fill_diagonal(dist_matrix, np.inf)
    sparse_dist = csr_matrix(dist_matrix)
    graph = DiGraph()
    rows, cols = sparse_dist.nonzero()
    for i, j in zip(rows, cols):
        weight = sparse_dist[i, j]
        if np.isfinite(weight) and weight != np.inf:
            graph.add_edge(i, j, weight=sparse_dist[i, j])
    mst = minimum_spanning_arborescence(graph)
    mst_matrix = np.zeros_like(dist_matrix, dtype=np.int32)
    for u, v, data in mst.edges(data=True):
        mst_matrix[u, v] = 1 # u -> v
    
    print(mst_matrix)
    print(texts)
    
    # patch description from VLM
    
    # constructing scene graph
    scene_graph = SceneGraph()
    for idx in range(node_num):
        scene_graph.add_node(label=texts[idx],
                             center=(lons[idx], lats[idx]),
                             description=None,
                             mask=original_masks[idx]
                             )
    for i in range(node_num):
        for j in range(node_num):
            if mst_matrix[i][j] == 1:
                scene_graph.add_edges(i, j)
    
    scene_description = scene_graph.get_text()
    print(scene_description)
    
    # visualize scene_graph
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    existing_texts = []
    for idx in range(node_num):
        x_pixel, y_pixel = x_center_original[idx], y_center_original[idx]
        draw.ellipse([x_pixel - 10, y_pixel - 10, x_pixel + 10, y_pixel + 10], outline="black", fill="yellow")
        draw.text((x_pixel + 5, y_pixel + 5), f'Node {texts[idx]}', fill="black")
    for edge in scene_graph.edges:
        source_pos = (x_center_original[edge.source], y_center_original[edge.source])
        target_pos = (x_center_original[edge.target], y_center_original[edge.target])
        draw.line([source_pos, target_pos], fill="red", width=2)
        
        mid_point = ((x_center_original[edge.source] + x_center_original[edge.target]) / 2, (y_center_original[edge.source] + y_center_original[edge.target]) / 2)
        for existing_text in existing_texts:
            if abs(existing_text[0] - mid_point[0]) < 5 and abs(existing_text[1] - mid_point[1]) < 5:
                mid_point = (mid_point[0] - 10, mid_point[1])
                break
        existing_texts.append(mid_point)
        draw.text(mid_point, edge.direction, fill="red")
        
    graph_path = os.path.join(save_folder, f"graph.png")
    img.save(graph_path, "PNG")
    
    # reasoning from LLM for scene graph

if __name__ == "__main__":
    img_path = "./data/Google/Changsha/112.922586488_28.164847797.tif"
    img = Image.open(img_path)
    img = np.array(img)
    navigator(img, 0,0,0,0)