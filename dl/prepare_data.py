# dl/prepare_data.py
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Locate the extracted JSON file relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "data" / "raw" / "InfraredSolarModules"
OUT = PROJECT_ROOT / "data" / "processed" / "ism.npz"
SEED = 42

def main():
    print("Loading metadata...")
    meta_path = ROOT / "module_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}. Check extraction path.")
        
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    # 1. Define strict alphabetical class mapping
    classes = sorted({item["anomaly_class"] for item in meta.values()})
    class_to_index = {name: i for i, name in enumerate(classes)}
    
    # 2. Sort IDs numerically (not lexicographically) to ensure deterministic ordering
    ids = sorted(meta.keys(), key=int)

    images, labels = [], []
    
    print("Decoding 20,000 JPEGs into NumPy arrays...")
    for image_id in ids:
        relative_path = meta[image_id]["image_filepath"]
        path = ROOT / relative_path
        
        # Convert to Grayscale ('L') and cast to uint8
        image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        
        # Ensure dimensional consistency: PIL size (24, 40) -> NumPy shape (H=40, W=24)
        if image.shape != (40, 24):
            raise ValueError(f"Unexpected image shape {image.shape} at {path}")
            
        images.append(image)
        labels.append(class_to_index[meta[image_id]["anomaly_class"]])

    # 3. Stack into continuous memory blocks
    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)
    idx = np.arange(len(y))
    
    print("Applying Stratified Split (70/15/15)...")
    # First split: 70% Train, 30% Temporary
    tr, temp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    # Second split: Divide the 30% Temporary equally into 15% Val and 15% Test
    va, te = train_test_split(temp_idx, test_size=0.50, random_state=SEED, stratify=y[temp_idx])

    # 4. Strict sanity checks to prevent data leakage
    assert len(set(tr) & set(va)) == 0, "Data leakage between Train and Val!"
    assert len(set(tr) & set(te)) == 0, "Data leakage between Train and Test!"
    assert len(set(va) & set(te)) == 0, "Data leakage between Val and Test!"
    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, tr=tr, va=va, te=te, classes=np.asarray(classes))
    
    print(f"\n--- Packaged Successfully to {OUT} ---")
    print(f"X Tensor Shape: {X.shape}")
    print(f"Split sizes -> Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
    
    print("\nClass Distribution (checking stratification):")
    counts = np.bincount(y)
    for class_name, count in zip(classes, counts):
        print(f"{class_name.rjust(18)}: {count}")

if __name__ == "__main__":
    main()