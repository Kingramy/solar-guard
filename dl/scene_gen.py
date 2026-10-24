# dl/scene_gen.py
import json
from pathlib import Path
import numpy as np
import cv2

def create_synthetic_scene(X_test, y_test, classes, scene_id, config):
    """Constructs a 4x6 grid (1016x424) from isolated 40x24 crops."""
    # Geometry constants
    rows, cols = 4, 6
    crop_h, crop_w = 40, 24
    scale = 4
    mod_w, mod_h = crop_w * scale, crop_h * scale # 96x160
    gap, margin = 8, 8
    img_w = 2 * margin + cols * mod_w + (cols - 1) * gap # 1016
    img_h = 2 * margin + rows * mod_h + (rows - 1) * gap # 424
    
    img = np.full((img_h, img_w), 20, dtype=np.uint8)
    temp = np.full((img_h, img_w), 25.0, dtype=np.float32)
    gt_data = []

    # Temperature mapping assumptions per class
    temp_ranges = {
        "No-Anomaly": (38.0, 44.0),
        "Soiling": (45.0, 54.0),
        "Offline-Module": (60.0, 78.0),
        "Diode": (60.0, 75.0),
        "Cell": (70.0, 98.0),
        "Shadowing": (45.0, 58.0),
        "Vegetation": (45.0, 58.0),
        "Cracking": (50.0, 65.0),
        "Diode-Multi": (65.0, 80.0),
        "Hot-Spot": (80.0, 110.0),
    }

    # Fetch available indices by class
    indices_by_class = {c: np.where(y_test == i)[0] for i, c in enumerate(classes)}

    for r in range(rows):
        for c in range(cols):
            x = margin + c * (mod_w + gap)
            y = margin + r * (mod_h + gap)
            
            # Select the class based on scene configuration
            target_class = "No-Anomaly"
            override_temp = None
            
            if scene_id == "scene_critical_01":
                if r == 2 and c == 4: target_class = "Cell"
                elif r == 1 and c == 1: target_class = "Diode"
                elif r == 3 and c == 2: target_class = "Soiling"
                # Demo Step 5: Explicit safety-override injection
                elif r == 0 and c == 5: 
                    target_class = "No-Anomaly"
                    override_temp = 95.0
            elif scene_id == "scene_dusty_03":
                if (r + c) % 3 == 0: target_class = "Soiling"
            elif scene_id == "scene_batch_04":
                if r == 1: target_class = "Offline-Module"
                if r == 2 and c in [0, 1]: target_class = "Diode"

            # Sample a crop
            idx = np.random.choice(indices_by_class.get(target_class, indices_by_class["No-Anomaly"]))
            crop = X_test[idx] # (40, 24)
            
            # Upscale and place in image canvas
            resized_crop = cv2.resize(crop, (mod_w, mod_h), interpolation=cv2.INTER_NEAREST)
            img[y:y+mod_h, x:x+mod_w] = resized_crop
            
            # Synthesize thermal map
            t_min, t_max = temp_ranges.get(target_class, (40.0, 50.0))
            base_temp = np.random.uniform(t_min, t_max)
            
            # Map pixel intensity to temperature variation
            norm_crop = resized_crop.astype(np.float32) / 255.0
            module_temp = (norm_crop * 10.0) + (base_temp - 5.0)
            temp[y:y+mod_h, x:x+mod_w] = np.clip(module_temp, 36.0, 150.0)
            
            # Inject localized hotspot if requested (Demo Step 5)
            if override_temp is not None:
                cy, cx = mod_h // 2, mod_w // 2
                temp[y+cy-8:y+cy+8, x+cx-12:x+cx+12] = override_temp
                
            gt_data.append({
                "row": r, "col": c, 
                "bbox": [int(x), int(y), int(mod_w), int(mod_h)],
                "true_class": target_class,
                "injected_override": (override_temp is not None)
            })

    return img, temp, gt_data

def main():
    np.random.seed(42) # Pin seed for reproducible scenes
    data_path = Path("data/processed/ism.npz")
    out_dir = Path("demo/scenes")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading test split for scene generation...")
    with np.load(data_path, allow_pickle=True) as data:
        X_test, y_test = data["X"][data["te"]], data["y"][data["te"]]
        classes = data["classes"]

    scenes = ["scene_critical_01", "scene_healthy_02", "scene_dusty_03", "scene_batch_04", "scene_mixed_05"]
    
    for sid in scenes:
        img, temp, gt = create_synthetic_scene(X_test, y_test, classes, sid, None)
        np.savez_compressed(out_dir / f"{sid}.npz", img=img, temp=temp)
        
        with open(out_dir / f"{sid}_gt.json", "w") as f:
            json.dump(gt, f, indent=2)
            
        print(f"Generated {sid} (1016x424)")

if __name__ == "__main__":
    main()