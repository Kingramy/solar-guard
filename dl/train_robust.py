# dl/train_robust.py
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from dl.model import ThermalCNN

class RobustThermalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean: float, std: float, augment: bool = False):
        self.X = torch.from_numpy(X).float().unsqueeze(1) 
        self.X = (self.X - mean) / std
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx].clone()
        
        if self.augment:
            # 1. Random Geometric Flips (Drone flying opposite directions)
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, [1]) # Vertical flip
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, [2]) # Horizontal flip
                
            # 2. Random Gaussian Sensor Noise (Simulating microbolometer static)
            if torch.rand(1).item() < 0.5:
                # Add noise relative to the normalized image standard deviation
                noise = torch.randn_like(img) * (torch.rand(1).item() * 0.15)
                img = img + noise
                
            # 3. Random Contrast Jitter (Simulating uncalibrated camera gain)
            if torch.rand(1).item() < 0.5:
                alpha = 0.8 + torch.rand(1).item() * 0.4
                img = img * alpha

        return img, self.y[idx]

def calculate_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=n_classes)
    max_count = np.max(counts)
    weights = np.sqrt(max_count / (counts + 1e-6))
    return torch.from_numpy(weights).float()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Physics-Augmented Training on: {device}")

    data_path = Path("data/processed/ism.npz")
    with np.load(data_path, allow_pickle=True) as data:
        X, y = data["X"], data["y"]
        tr, va = data["tr"], data["va"]
        classes = data["classes"]

    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    
    mean = float(np.mean(X_train))
    std = float(np.std(X_train)) + 1e-7 

    # Apply augmentations ONLY to the training set
    train_dataset = RobustThermalDataset(X_train, y_train, mean, std, augment=True)
    val_dataset = RobustThermalDataset(X_val, y_val, mean, std, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = ThermalCNN(n_classes=len(classes)).to(device)
    weights = calculate_class_weights(y_train, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    
    # Slightly higher weight decay to regularize the noisy augmentations
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1.5e-2)

    epochs = 40
    best_macro_f1 = 0.0
    checkpoint_dir = Path("models")
    
    print("Training Robust ThermalCNN...")
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.to(device))
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro")
        
        # Suppress epoch logs to keep terminal clean, just show every 5
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {tr_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "classes": classes,
                "mean": mean,
                "std": std,
                "val_macro_f1": best_macro_f1
            }, checkpoint_dir / "thermal_cnn.pt")

    print(f"\nRobust Model Saved! Best Val Macro-F1: {best_macro_f1:.4f}")

if __name__ == "__main__":
    main()