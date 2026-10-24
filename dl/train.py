# dl/train.py
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from dl.model import ThermalCNN

class ThermalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean: float, std: float):
        # Convert uint8 (0-255) to float32, add a channel dimension, and normalize
        self.X = torch.from_numpy(X).float().unsqueeze(1) 
        self.X = (self.X - mean) / std
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def calculate_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=n_classes)
    max_count = np.max(counts)
    # Square-root inverse-frequency
    weights = np.sqrt(max_count / (counts + 1e-6))
    return torch.from_numpy(weights).float()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Load the packed dataset directly into system RAM
    data_path = Path("data/processed/ism.npz")
    with np.load(data_path, allow_pickle=True) as data:
        X, y = data["X"], data["y"]
        tr, va = data["tr"], data["va"]
        classes = data["classes"]

    # 2. Prevent Data Leakage: Calculate statistics ONLY on the train split
    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    
    mean = float(np.mean(X_train))
    std = float(np.std(X_train)) + 1e-7 # Add epsilon to prevent division by zero

    # 3. Create PyTorch DataLoaders (batch_size=128 fits safely in VRAM)
    train_dataset = ThermalDataset(X_train, y_train, mean, std)
    val_dataset = ThermalDataset(X_val, y_val, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 4. Initialize the Model and Loss Function
    model = ThermalCNN(n_classes=len(classes)).to(device)
    
    weights = calculate_class_weights(y_train, len(classes)).to(device)
    # label_smoothing prevents the model from becoming overly confident in its predictions
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    
    # AdamW decoupling weight decay from the gradient update
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

    epochs = 40
    best_macro_f1 = 0.0
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)

    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # --- The Forward and Backward Pass ---
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()      # Wipe the gradients from the previous batch
            outputs = model(inputs)    # Forward pass: push data through the network
            loss = criterion(outputs, targets) # Calculate the mathematical error
            loss.backward()            # Backward pass: calculate gradients (calculus chain rule)
            optimizer.step()           # Update the weights
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad(): # Disable gradient tracking to save VRAM and compute
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # outputs are raw logits (Batch, 12). torch.argmax finds the index of the highest score
                preds = torch.argmax(outputs, dim=1) 
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Macro-F1 averages the F1 score of every class equally, ignoring class imbalance
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro")

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

        # Save the best model state
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "classes": classes,
                "mean": mean,
                "std": std,
                "val_macro_f1": best_macro_f1
            }, checkpoint_dir / "thermal_cnn.pt")
            print("  -> Checkpoint Saved!")

if __name__ == "__main__":
    main()