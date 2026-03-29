from SceneGraph_Generation.dinov3.loader import load_with_splitting, make_transform, SatelliteDataset
from torch.utils.data import DataLoader
import torch
from SceneGraph_Generation.Scene_graph_generator import EntityDetector
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import subprocess

def main():
    batch_size = 1
    epochs = 10
    split_ratio = 0.8
    lr = 1e-4
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_path = "./SceneGraph_Generation/models/dinov3_vitl16_pretrain_sat493m"
    image_folders = ["./data/Google/Changsha"]
    image_train_paths, label_train_paths, image_test_paths, label_test_paths = load_with_splitting(image_folders, split_ratio=split_ratio)
    image_paths = image_train_paths + image_test_paths
    label_paths = label_train_paths + label_test_paths
    
    transform = make_transform()
    
    train_dataset = SatelliteDataset(image_train_paths, label_train_paths, transform=transform, patch_size=(256, 256))
    test_dataset = SatelliteDataset(image_test_paths, label_test_paths, transform=transform, patch_size=(256, 256))
    inference_dataset = SatelliteDataset(image_paths, label_paths, transform=transform, patch_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    
    dino_dim = 1024
    
    model = EntityDetector(model_path, dino_dim).to(device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    log_dir='./log/detector'
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", log_dir, "--port", "6006"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("TensorBoard started at http://localhost:6006")

    time.sleep(3)
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, labels = batch
            labels = labels.to(device=device).float()
            if labels.ndim == 3: # [B, H, W] → [B, 1, H, W]
                labels = labels.unsqueeze(1)
            # paths = inputs['path']
            logits, _ = model(inputs)
            
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar("Train/Step_Loss", loss.item(), global_step)
            global_step += 1
        writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
        print(f"Epoch {epoch} Loss: {epoch_loss}")
    
    writer.close()
    tb_process.terminate()
    
    # Test the model
    model.eval()
    with torch.no_grad():
        total_loss = 0
        acc = 0
        total = 0
        for batch in tqdm(test_loader, desc="Test"):
            inputs, labels = batch
            labels = labels.to(device=device).float()
            if labels.ndim == 3:
                labels = labels.unsqueeze(1)
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5)
            labels_bool = (labels > 0.5)
            acc += (preds == labels_bool).sum().item()
            total += preds.numel()
        print(f"Test Loss: {total_loss}, Accuracy: {acc/total}")
        
    # Inference and saving
    save_path = "./SceneGraph_Generation/models/Entity_Detector/model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path) # save model
    
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Saving"):
            inputs, _ = batch
            paths = inputs['path']
            _, features = model(inputs)
            features = features.detach().cpu().numpy()
            for i, path in enumerate(paths):
                p = Path(path)
                new_path = p.with_suffix(".npy")
                folder = new_path.parent.name
                filename = new_path.name
                save_dir = os.path.join('./outputs', folder)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, filename)
                feature = features[i]
                np.save(save_path, feature)
    
if __name__ == "__main__":
    main()