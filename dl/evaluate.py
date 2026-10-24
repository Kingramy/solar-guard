# dl/evaluate.py
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

from dl.model import ThermalCNN
from dl.train import ThermalDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 1. Load Data and Checkpoint
    data_path = Path("data/processed/ism.npz")
    ckpt_path = Path("models/thermal_cnn.pt")
    mapping_path = Path("contracts/class_mapping.json")

    with np.load(data_path, allow_pickle=True) as data:
        X_test, y_test = data["X"][data["te"]], data["y"][data["te"]]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    mean, std = ckpt["mean"], ckpt["std"]
    
    with open(mapping_path, "r") as f:
        class_to_agent = json.load(f)

    # 2. Run Inference on the Held-Out Test Set
    test_dataset = ThermalDataset(X_test, y_test, mean, std)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = ThermalCNN(n_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 3. Calculate 12-Class Metrics
    acc_12 = accuracy_score(all_targets, all_preds)
    f1_12_macro = f1_score(all_targets, all_preds, average="macro")

    # 4. Calculate Binary Metrics (Healthy vs Faulty)
    # "No-Anomaly" is the healthy baseline; everything else is faulty
    healthy_idx = list(classes).index("No-Anomaly")
    binary_targets = (all_targets != healthy_idx).astype(int)
    binary_preds = (all_preds != healthy_idx).astype(int)
    
    bin_acc = accuracy_score(binary_targets, binary_preds)
    bin_f1 = f1_score(binary_targets, binary_preds, pos_label=1)

    # 5. Calculate 8-Category Agent Metrics
    agent_targets = [class_to_agent[classes[t]] for t in all_targets]
    agent_preds = [class_to_agent[classes[p]] for p in all_preds]
    agent_acc = accuracy_score(agent_targets, agent_preds)
    agent_f1_macro = f1_score(agent_targets, agent_preds, average="macro")

    # 6. Save Reports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    metrics = {
        "test_size": len(all_targets),
        "accuracy_12_class": round(acc_12, 4),
        "macro_f1_12_class": round(f1_12_macro, 4),
        "accuracy_binary_faulty": round(bin_acc, 4),
        "f1_binary_faulty": round(bin_f1, 4),
        "accuracy_8_category_agent": round(agent_acc, 4),
        "macro_f1_8_category_agent": round(agent_f1_macro, 4)
    }

    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(reports_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))

    # 7. Generate and Save Confusion Matrix PNG
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="left")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("12-Class Thermal Confusion Matrix")
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center',
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    print(f"Metrics saved to {reports_dir}. 12-Class Accuracy: {acc_12:.4f}, Binary Faulty Accuracy: {bin_acc:.4f}")

if __name__ == "__main__":
    main()