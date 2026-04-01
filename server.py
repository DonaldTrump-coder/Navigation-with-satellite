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
import re
import ast
import math

class SceneGraphNavigator:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        dino_path = "./SceneGraph_Generation/models/dinov3_vitl16_pretrain_sat493m"
        dino_dim = 1024
        max_length = 8
        self.transform = make_transform()
        
        detector_path = "./SceneGraph_Generation/models/Entity_Detector/model.pt"
        self.detector = EntityDetector(dino_path, dino_dim).to(device=self.device)
        detector_state_dict = torch.load(
            detector_path,
            map_location="cpu"
        )
        expander_state_dict = {k.replace("expander.", ""): v 
                        for k, v in detector_state_dict.items() if k.startswith("expander.")}
        self.detector.expander_state_dict = expander_state_dict
        filtered_state_dict = {k: v for k, v in detector_state_dict.items() if not k.startswith("expander.")}
        self.detector.load_state_dict(filtered_state_dict, strict=False)
        self.detector.eval()
        self.detector.to(self.device)
        
        self.roi_size = (256, 256)
        cnn_ckpt = "./SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-RN50.pt"
        vit_ckpt = "./SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-ViT-B-32.pt"
        cnn_model, _, self.cnn_preprocess = open_clip.create_model_and_transforms('RN50', pretrained=cnn_ckpt)
        vit_model, _, self.vit_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=vit_ckpt)
        text_decoder_path = "./SceneGraph_Generation/models/GLM-OCR"
        self.text_decoder = load_ocr_model(text_decoder_path)
        self.text_processor = AutoProcessor.from_pretrained(text_decoder_path)
        
        generator_path = "./SceneGraph_Generation/models/Entity_Generator/model.pt"
        generator_state_dict = torch.load(
            generator_path,
            map_location="cpu"
        )
        
        self.fuser = Feature_Fuser(vector_dim=dino_dim, cnnmodel=cnn_model, vitmodel=vit_model).to(self.device)
        
        self.generator = Entity_Generator(vector_dim=dino_dim,
                             ocrmodel=self.text_decoder,
                             lora_config=None,
                             max_length=max_length
                             ).to(self.device)
        self.generator.load_state_dict(generator_state_dict)
        self.generator.eval()
        self.generator.inferring = True
        print("initialized!")
        
    def infer(self, img: np.ndarray, min_lon, max_lon, min_lat, max_lat):
        img = Image.fromarray(img)
        self.img = img
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        dataset = SatelliteDataset_infer(img, self.transform, (256, 256))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in dataloader:
            logits, features = self.detector(batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        preds = preds[0]
        mask = preds.squeeze(0).cpu().numpy()
        features = features[0].detach().cpu().numpy()
        filtered_masks = split_entities(mask) # [mask1, mask2 ...]
        self.node_num = len(filtered_masks)
        
        dataset = Patches_dataset_infer(features=features,
                                    img=img,
                                    masks=filtered_masks,
                                    cnn_preprocess=self.cnn_preprocess,
                                    vit_preprocess=self.vit_preprocess,
                                    roi_size=self.roi_size,
                                    )
        patch_batch_size = dataset.__len__()
        patch_dataloader = DataLoader(dataset, batch_size=patch_batch_size)
        for batch in patch_dataloader:
            image = batch["image"].to(self.device)
            entity_feature = batch["entity_feature"].to(self.device)
            entity_original = batch["entity_original"].to(self.device)
            self.fuser.eval()
            with torch.no_grad():
                fused_entity_features = self.fuser(entity_feature,
                                            entity_original,
                                            image)
                
        dataset = Patch_features_dataset_infer(fused_entity_features)
        dataloader = DataLoader(dataset, batch_size=patch_batch_size)
        for batch in dataloader:
            fused_entity_feature = batch.to(self.device)
            offsets, generated_ids = self.generator(fused_entity_feature)
        offsets = offsets.detach().cpu().numpy()
        generated_ids = generated_ids.detach().cpu()
        self.texts = self.text_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True) # texts list
        self.texts = [text.replace("\n", "").replace("_", "") for text in self.texts]
        
        self.x_center_original = []
        self.y_center_original = []
        self.descriptions = [] # list[str]
        self.original_masks = []
        self.original_bounds = []
        self.patches_origin = []
        self.lons = []
        self.lats = []
        self.dist_matrix = np.zeros((self.node_num, self.node_num), dtype=np.float32)
        self.original_width, self.original_height = img.size
        h, w = mask.shape
        h_scale, w_scale = self.original_height / h, self.original_width / w
        patch_height = 256
        self.save_folder = "./outputs/test"
        os.makedirs(self.save_folder, exist_ok=True)
        
        for idx, filtered_mask in enumerate(filtered_masks):
            # the idx-th patch
            patch_folder = os.path.join(self.save_folder, f"{idx:03d}")  # 000, 001, 002 ...
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
            self.x_center_original.append(x_center)
            self.y_center_original.append(y_center)
            
            transformed_mask = filtered_mask.astype(np.uint8)
            transformed_mask = cv2.resize(transformed_mask,
                                        (self.original_width, self.original_height),
                                        interpolation=cv2.INTER_NEAREST
                                        ) # [original_height, original_width]
            self.original_masks.append(transformed_mask)
            
            contours, _ = cv2.findContours(
                transformed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            boundary = np.vstack([c[:, 0, :] for c in contours])  # [points_num, 2]
            self.original_bounds.append(boundary)
            
            x_min_patch = int(np.floor(x_min * w_scale))
            x_max_patch = int(np.ceil(x_max * w_scale))
            y_min_patch = int(np.floor(y_min * h_scale))
            y_max_patch = int(np.ceil(y_max * h_scale))
            x_min_patch = max(0, x_min_patch)
            y_min_patch = max(0, y_min_patch)
            x_max_patch = min(self.original_width, x_max_patch)
            y_max_patch = min(self.original_height, y_max_patch)
            patch = img.crop((x_min_patch, y_min_patch, x_max_patch, y_max_patch))
            w, h = patch.size
            patch_width = int(w * (patch_height / h))
            patch = patch.resize((patch_width, patch_height), Image.BILINEAR)
            self.patches_origin.append(patch)
            
            lon, lat = pix2geo(x_center, y_center, min_lon, max_lon, min_lat, max_lat, self.original_width, self.original_height)
            self.lons.append(lon)
            self.lats.append(lat)
            
            patch_path = os.path.join(patch_folder, f"patch.png")
            patch.save(patch_path)
        
        mask_path = os.path.join(self.save_folder, "mask.png")
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(mask_path)
            
        # constructing relation (distance) matrix
        for i in range(self.node_num):
            center = np.array([self.x_center_original[i], self.y_center_original[i]])
            for j in range(self.node_num):
                if i == j:
                    continue
                boundary = self.original_bounds[j]
                distances = np.linalg.norm(boundary - center, axis=1)
                self.dist_matrix[i, j] = distances.min()
            
        # constructing connection
        np.fill_diagonal(self.dist_matrix, np.inf)
        sparse_dist = csr_matrix(self.dist_matrix)
        graph = DiGraph()
        rows, cols = sparse_dist.nonzero()
        for i, j in zip(rows, cols):
            weight = sparse_dist[i, j]
            if np.isfinite(weight) and weight != np.inf:
                graph.add_edge(i, j, weight=sparse_dist[i, j])
        mst = minimum_spanning_arborescence(graph)
        self.mst_matrix = np.zeros_like(self.dist_matrix, dtype=np.int32)
        for u, v, data in mst.edges(data=True):
            self.mst_matrix[u, v] = 1
        
        print(self.mst_matrix)
        print(self.texts)
        
    def get_patched(self):
        return self.patches_origin, self.texts
    
    def set_descriptions(self, descriptions):
        self.descriptions = descriptions
        
    def get_scene_graph(self):
        # constructing scene graph
        self.scene_graph = SceneGraph()
        for idx in range(self.node_num):
            self.scene_graph.add_node(label=self.texts[idx],
                                center=(self.lons[idx], self.lats[idx]),
                                pix_center=(self.x_center_original[idx], self.y_center_original[idx]),
                                description=self.descriptions[idx],
                                mask=self.original_masks[idx]
                                )
        for i in range(self.node_num):
            for j in range(self.node_num):
                if self.mst_matrix[i][j] == 1:
                    self.scene_graph.add_edges(i, j)
        
        scene_description = self.scene_graph.get_text()
        print(scene_description)
        
        # visualize scene_graph
        img = self.img.convert("RGBA")
        draw = ImageDraw.Draw(img)
        existing_texts = []
        for idx in range(self.node_num):
            x_pixel, y_pixel = self.x_center_original[idx], self.y_center_original[idx]
            draw.ellipse([x_pixel - 10, y_pixel - 10, x_pixel + 10, y_pixel + 10], outline="black", fill="yellow")
            draw.text((x_pixel + 5, y_pixel + 5), f'Node {self.texts[idx]}', fill="black")
        for edge in self.scene_graph.edges:
            source_pos = (self.x_center_original[edge.source], self.y_center_original[edge.source])
            target_pos = (self.x_center_original[edge.target], self.y_center_original[edge.target])
            draw.line([source_pos, target_pos], fill="red", width=2)
            
            mid_point = ((self.x_center_original[edge.source] + self.x_center_original[edge.target]) / 2, (self.y_center_original[edge.source] + self.y_center_original[edge.target]) / 2)
            for existing_text in existing_texts:
                if abs(existing_text[0] - mid_point[0]) < 5 and abs(existing_text[1] - mid_point[1]) < 5:
                    mid_point = (mid_point[0] - 100, mid_point[1])
                    break
            existing_texts.append(mid_point)
            draw.text(mid_point, edge.direction, fill="red")
            
        graph_path = os.path.join(self.save_folder, "graph.png")
        img.save(graph_path, "PNG")
        
        return scene_description
    
    def get_navigation_points(self,
                              llm_answers,
                              start_point, # pix(x, y)
                              survey_areas # [[idxs], [idxs],...]
                              ):
        # reasoning from LLM for scene graph
        # Params: [[max_interval(m), expand_rate, flight_interval(m), flight_speed(m/s)],[max_interval(m), expand_rate, flight_interval(m), flight_speed(m/s)],...]
        # Routes: [[idx1, idx2...],[idx2, idx3...],...]
        match = re.search(r"Answer:\s*(.*)", llm_answers, re.DOTALL)
        if match:
            answer_text = match.group(1)
            params_match = re.search(r"Params:\s*(\[\[.*?\]\])", answer_text, re.DOTALL)
            routes_match = re.search(r"Routes:\s*(\[\[.*?\]\])", answer_text, re.DOTALL)
            
            params = None
            routes = None
            
            if params_match:
                params_text = params_match.group(1)
                params = ast.literal_eval(params_text)
            
            if routes_match:
                routes_text = routes_match.group(1)
                routes = ast.literal_eval(routes_text)
        else:
            return None
        
        traj_points = [Traj_Point("traj", start_point[0], start_point[1])]
        task_num = len(params)
        for i in range(task_num):
            max_interval = params[i][0]
            expand_rate = params[i][1]
            flight_interval = params[i][2]
            
            _, height = self.img.size
            lat_distance = self.max_lat - self.min_lat
            lat_res = lat_distance / height # deg / pixel
            deg_to_m = 111000
            lat_res *= deg_to_m # meter / pixel
            max_interval /= lat_res # pixel
            flight_interval /= lat_res # pixel
            
            points_num = len(routes[i])
            objects = survey_areas[i] # list[idx1, idx2, ...]
            if i == 0:
                for j in range(points_num - 1):
                    point = routes[i][j] # idx
                    center = self.scene_graph.nodes[point].pix_center
                    traj_points.append(Traj_Point("traj", center[0], center[1]))
            else:
                for j in range(1, points_num - 1):
                    point = routes[i][j]  # idx
                    center = self.scene_graph.nodes[point].pix_center
                    traj_points.append(Traj_Point("traj", center[0], center[1]))
            if points_num == 1:
                front_id = None
            else:
                front_id = routes[i][points_num - 2]
            if i + 1 == task_num:
                next_id = None
            else:
                next_id = routes[i+1][1]
            flight_points = self.scene_graph.get_flight_points(start=start_point,
                                               front_id=front_id,
                                               object_ids=objects,
                                               next_id=next_id,
                                               max_interval=max_interval,
                                               expand_rate=expand_rate
                                               )
            flight_points = insert_points(flight_points, flight_interval)
            for point in flight_points:
                traj_points.append(Traj_Point("survey", point[0], point[1]))
        
        traj_points.append(Traj_Point("traj", start_point[0], start_point[1]))
        
        img = self.img.convert("RGBA")
        draw = ImageDraw.Draw(img)
        point_color = (255, 0, 0)
        point_radius = 2
        for point in traj_points:
            draw.ellipse(
                [(point.x - point_radius, point.y - point_radius), 
                (point.x + point_radius, point.y + point_radius)], 
                fill=point_color
            )
        traj_path = os.path.join(self.save_folder, "traj.png")
        img.save(traj_path, "PNG")
        
        return traj_points # pixel(x, y)
                
def insert_points(flight_points, interval):
    new_points = []
    for i in range(len(flight_points) - 1):
        start_point = flight_points[i]
        end_point = flight_points[i + 1]
        distance = np.linalg.norm(end_point - start_point)
        new_points.append(start_point)
        if distance > interval:
            num_new_points = math.ceil(distance / interval) - 1
            for j in range(1, num_new_points + 1):
                t = j * interval / distance
                new_x = start_point[0] + t * (end_point[0] - start_point[0])
                new_y = start_point[1] + t * (end_point[1] - start_point[1])
                new_points.append(np.array([new_x, new_y]))
    new_points.append(flight_points[-1])
    return new_points

class Traj_Point:
    def __init__(self, kind, x, y):
        if kind == "survey":
            self.kind = "survey"
        elif kind == "traj":
            self.kind = "traj"
        self.x = x
        self.y = y
    
    def to_dict(self):
        return {
            "kind": self.kind,
            "x": float(self.x),
            "y": float(self.y)
        }
    
from fastapi import FastAPI, Response, Request
from pydantic import BaseModel
import base64
from io import BytesIO
import pickle

app = FastAPI()
navigator = SceneGraphNavigator()

class InferRequest(BaseModel):
    img: str  # in base64
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    
class GetTrajectoryRequest(BaseModel):
    llm_answers: str
    start_point: tuple
    survey_areas: list

@app.post("/infer")
async def infer(request: InferRequest):
    img_bytes = base64.b64decode(request.img)
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img)
    navigator.infer(img, request.min_lon, request.max_lon, request.min_lat, request.max_lat)
    return {"message": "Received"}

@app.get("/get_patches")
def get_patches():
    patches, texts = navigator.get_patched()
    data = pickle.dumps({
        "patches": patches,
        "texts": texts
    })
    return Response(content=data, media_type="application/octet-stream")

@app.post("/set_descriptions")
async def set_descriptions(request: Request):
    data = await request.body()
    descriptions = pickle.loads(data)
    navigator.set_descriptions(descriptions)
    return {"message": "Descriptions set successfully"}

@app.post("/scene_description")
async def send_scene_description():
    scene_description = navigator.get_scene_graph()
    return {"message": "Scene description sent successfully", "scene_description": scene_description}

@app.get("/get_trajectory")
async def get_trajectory(request: GetTrajectoryRequest):
    llm_answers = request.llm_answers
    start_point = request.start_point
    survey_areas = request.survey_areas
    traj_points = navigator.get_navigation_points(llm_answers, start_point, survey_areas)
    return {"traj_points": [p.to_dict() for p in traj_points]}