# dl/infer.py
import json
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from dl.model import ThermalCNN
from dl.thermal import segment_modules, compute_scene_baseline, find_hot_cell_bbox

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Load Model
    ckpt_path = Path("models/thermal_cnn.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    mean, std = ckpt["mean"], ckpt["std"]
    
    model = ThermalCNN(n_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load mapping
    with open("contracts/class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    scenes_dir = Path("demo/scenes")
    
    for npz_file in scenes_dir.glob("*.npz"):
        scene_id = npz_file.stem
        data = np.load(npz_file)
        img, temp_map = data["img"], data["temp"]
        
        h, w = img.shape
        
        # 1. Deterministic Localization & Baseline
        boxes = segment_modules(temp_map, threshold_c=32.0)
        scene_baseline = compute_scene_baseline(temp_map, boxes)
        
        detections = []
        n_anomalies = 0
        n_alerts = 0
        
        # Display image setup (False-color Inferno)
        display_img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        
        for i, bbox in enumerate(boxes):
            bx, by, bw, bh = bbox
            
            # 2. Extract and format crop for CNN
            crop = img[by:by+bh, bx:bx+bw]
            crop_resized = cv2.resize(crop, (24, 40), interpolation=cv2.INTER_LINEAR)
            
            tensor_crop = torch.from_numpy(crop_resized).float().unsqueeze(0).unsqueeze(0)
            tensor_crop = (tensor_crop - mean) / std
            
            # 3. CNN Classification
            with torch.no_grad():
                logits = model(tensor_crop.to(device))
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            top_indices = np.argsort(probs)[::-1]
            pred_class = classes[top_indices[0]]
            confidence = float(probs[top_indices[0]])
            top3 = [[classes[idx], float(probs[idx])] for idx in top_indices[:3]]
            
            agent_cat = class_mapping[pred_class]
            
            # 4. Thermal Safety Guardrails
            mod_max_temp = round(float(np.max(temp_map[by:by+bh, bx:bx+bw])), 2)
            delta_t = round(mod_max_temp - scene_baseline, 2)
            
            # Heuristic hotspot localization
            hot_cell = find_hot_cell_bbox(temp_map, bbox)
            
            # Flag Logic (Rule 1 & 2 from manual)
            temp_triggered = (mod_max_temp >= 90.0) or (delta_t >= 40.0)
            
            if agent_cat != "none" and temp_triggered:
                flag = "model_and_temperature"
                n_anomalies += 1
                n_alerts += 1
            elif agent_cat != "none" and not temp_triggered:
                flag = "model"
                n_anomalies += 1
            elif agent_cat == "none" and temp_triggered:
                flag = "temperature"
                n_alerts += 1
            else:
                flag = None
                
            if flag:
                det = {
                    "id": f"D-{i:04d}",
                    "row": int(by // 100), "col": int(bx // 160), # Approximated for JSON
                    "bbox": [int(v) for v in bbox],
                    "hot_cell_bbox": [int(v) for v in hot_cell] if hot_cell is not None else None,
                    "ml_class": str(pred_class),
                    "agent_category": str(agent_cat),
                    "confidence": round(float(confidence), 3),
                    "top3": [[str(c), round(float(p), 4)] for c, p in top3],
                    "max_temp_c": float(mod_max_temp),
                    "delta_t_c": float(delta_t),
                    "flag_reason": flag
                }
                detections.append(det)
                
                # Draw on annotated image
                color = (0, 0, 255) if temp_triggered else (0, 165, 255)
                cv2.rectangle(display_img, (bx, by), (bx+bw, by+bh), color, 2)
                cv2.putText(display_img, f"{pred_class} {mod_max_temp}C", (bx, by-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 5. Build Final JSON Contract
        contract = {
            "schema_version": "1.0",
            "scene_id": scene_id,
            "site": {"name": "Demo Solar Park (fictional)", "sector": "Sector 4", "array": "B"},
            "image_size": {"width": w, "height": h},
            "temperature_source": "synthetic",
            "baseline_method": "scene_module_median",
            "scene_baseline_temp_c": scene_baseline,
            "detections": detections,
            "summary": {
                "n_modules": 24,
                "n_predicted_anomalies": n_anomalies,
                "n_temperature_alerts": n_alerts
            }
        }
        
        with open(scenes_dir / f"{scene_id}.json", "w") as f:
            json.dump(contract, f, indent=2)
            
        cv2.imwrite(str(scenes_dir / f"{scene_id}.png"), display_img)
        print(f"Inference complete for {scene_id}. Found {len(detections)} actionable panels.")

if __name__ == "__main__":
    main()