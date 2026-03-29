import torch, open_clip
import os
import random
from SceneGraph_Generation.datasets import Patches_dataset, Patches_relation_dataset, Patch_features_dataset
from transformers import AutoProcessor
from SceneGraph_Generation.modules.Languagemodels.GLMOCR import load_ocr_model
from torch.utils.data import DataLoader
import itertools
from SceneGraph_Generation.modules.Entity_Generation.Entity_Generator import Entity_Generator, Feature_Fuser
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, TaskType
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import subprocess
import time

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
    "connections": [[1, 2], [1, 3], ...]
}
"""


def main():
    batch_size = 8
    train_ratio = 0.8
    roi_size = (256, 256)
    lr=1e-4
    max_length = 8 # max token length of entity labels
    epochs = 10
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    cnn_ckpt = "./SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-RN50.pt"
    vit_ckpt = "./SceneGraph_Generation/models/RemoteCLIP/RemoteCLIP-ViT-B-32.pt"
    cnn_model, _, cnn_preprocess = open_clip.create_model_and_transforms('RN50', pretrained=cnn_ckpt)
    vit_model, _, vit_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=vit_ckpt)
    
    text_decoder_path = "./SceneGraph_Generation/models/GLM-OCR"
    text_decoder = load_ocr_model(text_decoder_path)
    text_processor = AutoProcessor.from_pretrained(text_decoder_path)
    
    features_folder = "./outputs"
    data_folder = "./data/Google"
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules = ["q_proj", "v_proj"],
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    dino_dim = 1024
    
    log_dir="./log/generator"
    writer = SummaryWriter(log_dir=log_dir)
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", log_dir, "--port", "6006"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("TensorBoard started at http://localhost:6006")

    time.sleep(3)
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    
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
    model = Entity_Generator(vector_dim=dino_dim,
                             ocrmodel=text_decoder,
                             lora_config = lora_config
                             ).to(device)
    model.inferring = False # training and testing mode
    encoder = Feature_Fuser(vector_dim=dino_dim, cnnmodel=cnn_model, vitmodel=vit_model).to(device)
    
    # training
    logits_loss = offset_loss = relation_loss = torch.tensor(0.0)
    for epoch in range(epochs):
        epoch_loss = 0
        if epoch == 0: # first stage
            model.set_train_stage("stage1")
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
            print("Training stage 1")
        elif epoch == int(0.6 * epochs): # second stage
            model.set_train_stage("stage2")
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = lr * 0.1)
            print("Training stage 2")
        for scene in tqdm(train_samples, desc=f"Epoch {epoch}"):
            npy_path, tif_path, json_path = scene
            encoder_dataset = Patches_dataset(npy_path,
                                            tif_path,
                                            json_path,
                                            cnn_preprocess,
                                            vit_preprocess,
                                            roi_size
                                            )
            shapes = encoder_dataset.get_shapes()
            patch_batch_size = encoder_dataset.__len__()
            patch_dataloader = DataLoader(encoder_dataset, batch_size=patch_batch_size)
            for batch in patch_dataloader:
                image = batch["image"].to(device)
                entity_feature = batch["entity_feature"].to(device)
                entity_original = batch["entity_original"].to(device)
                with torch.no_grad():
                    fused_entity_features = encoder(entity_feature,
                                                    entity_original,
                                                    image)
            
            train_dataset = Patch_features_dataset(fused_entity_features,
                                                shapes,
                                                text_processor,
                                                max_length = max_length
                                                )
            train_relation_dataset = Patches_relation_dataset(json_path,
                                                            fused_entity_features,
                                                            shapes
                                                            )
            
            patch_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            patch_relation_loader = DataLoader(train_relation_dataset, batch_size=batch_size, shuffle=True)
            num_steps = max(len(patch_loader), len(patch_relation_loader))
            patch_loader = itertools.cycle(patch_loader)
            patch_relation_loader = itertools.cycle(patch_relation_loader)
            
            for step in range(num_steps):
                optimizer.zero_grad()
                if step % 2 == 0:
                    model.relation = False
                    batch = next(patch_loader)
                    fused_entity_features = batch["fused_entity_features"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_masks = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    offset_labels = batch["offset"].to(device)
                    logits, offsets = model(fused_entity_features = fused_entity_features,
                                            input_ids = input_ids,
                                            attention_mask = attention_masks
                                            )
                    
                    # loss for logits
                    logits_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100
                    )
                    
                    # loss for offsets
                    offset_loss = F.l1_loss(offsets, offset_labels)
                    loss = logits_loss + offset_loss
                    writer.add_scalar("loss/logits", logits_loss.item(), global_step)
                    writer.add_scalar("loss/offset", offset_loss.item(), global_step)
                    
                    epoch_loss += logits_loss.item() + offset_loss.item()
                else:
                    model.relation = True
                    batch = next(patch_relation_loader)
                    fused_entity_features = batch["fused_entity_features"].to(device)
                    label = batch["label"].to(device).unsqueeze(-1) # [batch]
                    logits = model(fused_entity_features = fused_entity_features)
                    relation_loss = nn.BCEWithLogitsLoss()(logits, label.float())
                    writer.add_scalar("loss/relation", relation_loss.item(), global_step)
                    loss = relation_loss
                    
                    epoch_loss = relation_loss.item()
                    
                writer.add_scalar("loss/total", logits_loss.item() + offset_loss.item() + relation_loss.item(), global_step)
                
                loss.backward()
                optimizer.step()
                global_step += 1
                
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
                
    writer.close()
    tb_process.terminate()
    
    # testing
    model.eval()
    total_logits_loss = 0.0
    total_offset_loss = 0.0
    total_relation_loss = 0.0
    num_entity_batches = 0
    num_relation_batches = 0
    
    # save model
    model.lora_merge()
    save_path = "./SceneGraph_Generation/models/Entity_Generator/model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    with torch.no_grad():
        for scene in test_samples:
            npy_path, tif_path, json_path = scene
            encoder_dataset = Patches_dataset(npy_path,
                                              tif_path,
                                              json_path,
                                              cnn_preprocess,
                                              vit_preprocess,
                                              roi_size)
            shapes = encoder_dataset.get_shapes()
            patch_batch_size = encoder_dataset.__len__()
            patch_dataloader = DataLoader(encoder_dataset, batch_size=patch_batch_size)
            for batch in patch_dataloader:
                image = batch["image"].to(device)
                entity_feature = batch["entity_feature"].to(device)
                entity_original = batch["entity_original"].to(device)
                fused_entity_features = encoder(entity_feature,
                                                entity_original,
                                                image
                                                )
                
                test_dataset = Patch_features_dataset(fused_entity_features,
                                                      shapes,
                                                      text_processor,
                                                      max_length = max_length
                                                      )
                test_relation_dataset = Patches_relation_dataset(json_path,
                                                             fused_entity_features,
                                                             shapes)
                
                patch_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                patch_relation_loader = DataLoader(test_relation_dataset, batch_size=batch_size, shuffle=False)
                
                for batch in patch_loader:
                    model.relation = False
                    fused_entity_features = batch["fused_entity_features"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_masks = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    offset_labels = batch["offset"].to(device)
                    logits, offsets = model(fused_entity_features=fused_entity_features,
                                            input_ids=input_ids,
                                            attention_mask=attention_masks)
                    logits_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        ignore_index=-100
                    )
                    offset_loss = F.l1_loss(offsets, offset_labels)
                    total_logits_loss += logits_loss.item()
                    total_offset_loss += offset_loss.item()
                    num_entity_batches += 1
                    
                for batch in patch_relation_loader:
                    model.relation = True
                    fused_entity_features = batch["fused_entity_features"].to(device)
                    label = batch["label"].to(device)
                    logits = model(fused_entity_features=fused_entity_features)
                    relation_loss = nn.BCEWithLogitsLoss()(logits, label.float())
                    total_relation_loss += relation_loss.item()
                    num_relation_batches += 1
                    
    avg_logits_loss = total_logits_loss / max(num_entity_batches, 1)
    avg_offset_loss = total_offset_loss / max(num_entity_batches, 1)
    avg_relation_loss = total_relation_loss / max(num_relation_batches, 1)
    
if __name__ == '__main__':
    main()