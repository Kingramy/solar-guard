# dl/drone_infer.py
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from contracts.schema import SceneDetectionsContract
from dl.model import ThermalCNN
from dl.radiometric import rectify_panel_quad, normalize_crop_for_cnn, visual_to_proxy_celsius
from dl.drone_segmenter import PanelUNet, detect_tilted_panels, generate_wild_drone_frame


def assign_tilted_grid_indices(quads: List[np.ndarray]) -> List[Tuple[int, int, np.ndarray]]:
    """
    Projects the centers of tilted solar panel polygons onto the solar rack's
    principal angle so every panel receives its true (row, col) index regardless
    of drone camera rotation.
    """
    if not quads:
        return []

    # 1. Estimate the dominant rack angle from the top edge (p0 -> p1) of all panels
    angles = []
    centers = []
    heights = []
    for q in quads:
        dx = q[1, 0] - q[0, 0]
        dy = q[1, 1] - q[0, 1]
        angles.append(np.arctan2(dy, dx))
        centers.append(np.mean(q, axis=0))
        heights.append(np.linalg.norm(q[3] - q[0]))

    median_theta = float(np.median(angles))
    row_tol = float(np.median(heights)) * 0.55  # Half a panel height tolerance for row clustering

    cos_t, sin_t = np.cos(median_theta), np.sin(median_theta)

    # 2. Project each panel center onto (u: along row, v: across rows)
    projected = []
    for idx, (cx, cy) in enumerate(centers):
        u = cx * cos_t + cy * sin_t
        v = -cx * sin_t + cy * cos_t
        projected.append({"idx": idx, "u": u, "v": v, "quad": quads[idx]})

    # 3. Sort by v (top-to-bottom across rows) and cluster into rows
    projected.sort(key=lambda item: item["v"])
    rows_clusters: List[List[Dict[str, Any]]] = []
    for item in projected:
        if not rows_clusters:
            rows_clusters.append([item])
        else:
            mean_v = float(np.mean([x["v"] for x in rows_clusters[-1]]))
            if abs(item["v"] - mean_v) <= row_tol:
                rows_clusters[-1].append(item)
            else:
                rows_clusters.append([item])

    # 4. Inside each row, sort left-to-right by u and assign (row, col)
    indexed_panels: List[Tuple[int, int, np.ndarray]] = []
    for r_idx, cluster in enumerate(rows_clusters):
        cluster.sort(key=lambda item: item["u"])
        for c_idx, item in enumerate(cluster):
            indexed_panels.append((r_idx, c_idx, item["quad"]))

    return indexed_panels


def predict_with_tta(
    model_cnn: ThermalCNN,
    norm_crop_uint8: np.ndarray,
    mean: float,
    std: float,
    device: torch.device
) -> np.ndarray:
    """
    4-View Test-Time Augmentation (TTA):
    Evaluates [Original, Horizontal Flip, Vertical Flip, 180-Degree Flip] in a single batch
    and averages the Softmax probability distributions.
    """
    v1 = norm_crop_uint8
    v2 = np.fliplr(norm_crop_uint8).copy()
    v3 = np.flipud(norm_crop_uint8).copy()
    v4 = np.flipud(np.fliplr(norm_crop_uint8)).copy()

    batch_np = np.stack([v1, v2, v3, v4], axis=0).astype(np.float32)  # (4, 40, 24)
    batch_tensor = torch.from_numpy((batch_np - mean) / std).float().unsqueeze(1).to(device)  # (4, 1, 40, 24)

    with torch.no_grad():
        logits = model_cnn(batch_tensor)  # (4, 12)
        probs = F.softmax(logits, dim=1)  # (4, 12)
        avg_probs = torch.mean(probs, dim=0).cpu().numpy()  # (12,)

    return avg_probs


def analyze_drone_frame(
    visual_img: np.ndarray,
    temp_map: Optional[np.ndarray],
    scene_id: str,
    temp_source: str,
    model_unet: PanelUNet,
    model_cnn: ThermalCNN,
    ckpt_meta: Dict[str, Any],
    class_mapping: Dict[str, str],
    device: torch.device
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    End-to-End Wild Drone Frame Analyzer:
      1. If temp_map is None (user uploaded an 8-bit image), estimates proxy temperature via CLAHE.
      2. Segments tilted panel cores with PanelUNet and recovers 4-corner polygons.
      3. Rectifies each panel via 3x3 Homography (excluding all hot desert sand).
      4. Computes classifier-independent scene_module_median from rectified interiors.
      5. Classifies each panel with 4-View TTA ThermalCNN + applies thermal safety rules.
      6. Validates the resulting dict against SceneDetectionsContract.
    """
    if temp_map is None:
        visual_img, temp_map = visual_to_proxy_celsius(visual_img)
        temp_source = "estimated_visual"

    h, w = visual_img.shape[:2]
    classes = list(ckpt_meta["classes"])
    mean, std = float(ckpt_meta["mean"]), float(ckpt_meta["std"])

    # 1. Detect tilted panels using U-Net
    raw_quads, _ = detect_tilted_panels(visual_img, model_unet, device)
    indexed_quads = assign_tilted_grid_indices(raw_quads)

    if not indexed_quads:
        raise ValueError("No solar panels were detected in the uploaded frame.")

    # 2. Rectify every panel via Homography and collect pure interior thermal stats
    rectified_records = []
    mod_medians = []
    for r_idx, c_idx, quad in indexed_quads:
        rect_img, rect_temp = rectify_panel_quad(visual_img, temp_map, quad, out_w=24, out_h=40)
        mod_medians.append(float(np.median(rect_temp)))
        rectified_records.append((r_idx, c_idx, quad, rect_img, rect_temp))

    # 3. Classifier-Independent Baseline across all detected panel interiors
    scene_baseline = round(float(np.median(mod_medians)), 2)

    # Prepare false-color annotated visual output
    if visual_img.ndim == 3 and visual_img.shape[2] == 3:
        display_img = visual_img.copy()
    else:
        display_img = cv2.applyColorMap(visual_img.astype(np.uint8), cv2.COLORMAP_INFERNO)
    detections = []
    n_anomalies = 0
    n_alerts = 0

    for i, (r_idx, c_idx, quad, rect_img, rect_temp) in enumerate(rectified_records):
        # Domain-adapt the rectified crop and classify with 4-View TTA
        adapted_crop = normalize_crop_for_cnn(rect_img, rect_temp)
        probs = predict_with_tta(model_cnn, adapted_crop, mean, std, device)

        top_indices = np.argsort(probs)[::-1]
        pred_class = str(classes[top_indices[0]])
        confidence = float(probs[top_indices[0]])
        top3 = [[str(classes[idx]), round(float(probs[idx]), 4)] for idx in top_indices[:3]]
        agent_cat = class_mapping[pred_class]

        mod_max_temp = round(float(np.max(rect_temp)), 2)
        delta_t = round(mod_max_temp - scene_baseline, 2)

        # Axis-aligned bounding box enclosing the tilted polygon (for schema compatibility)
        xs, ys = quad[:, 0], quad[:, 1]
        bx = int(max(0, np.min(xs)))
        by = int(max(0, np.min(ys)))
        bw = int(min(w - bx, np.max(xs) - bx))
        bh = int(min(h - by, np.max(ys) - by))
        bbox = [bx, by, max(1, bw), max(1, bh)]
        poly_int = quad.astype(int).tolist()

        # Draw subtle green outline on all detected panels so the operator sees the U-Net grid
        cv2.polylines(display_img, [quad.astype(np.int32)], isClosed=True, color=(80, 200, 80), thickness=1)

        # Evaluate deterministic flagging rules
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
                "row": int(r_idx),
                "col": int(c_idx),
                "bbox": bbox,
                "polygon": poly_int,
                "hot_cell_bbox": None,
                "ml_class": pred_class,
                "agent_category": agent_cat,
                "confidence": round(confidence, 3),
                "top3": top3,
                "max_temp_c": mod_max_temp,
                "delta_t_c": delta_t,
                "flag_reason": flag
            }
            detections.append(det)

            # Highlight flagged tilted polygon in Red (thermal alert) or Orange (model fault)
            color = (0, 0, 255) if temp_triggered else (0, 165, 255)
            cv2.polylines(display_img, [quad.astype(np.int32)], isClosed=True, color=color, thickness=2)
            cv2.putText(
                display_img, f"R{r_idx}C{c_idx}:{pred_class[:5]} {mod_max_temp:.0f}C",
                (bx, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
            )

    contract = {
        "schema_version": "1.0",
        "scene_id": scene_id,
        "site": {"name": "Sahara Drone Inspection Sector", "sector": "Sector 7 (Aerial)", "array": "Drone-Wing-A"},
        "image_size": {"width": int(w), "height": int(h)},
        "temperature_source": temp_source,
        "baseline_method": "scene_module_median",
        "scene_baseline_temp_c": scene_baseline,
        "detections": detections,
        "summary": {
            "n_modules": len(indexed_quads),
            "n_predicted_anomalies": n_anomalies,
            "n_temperature_alerts": n_alerts
        }
    }

    # Strict Pydantic schema validation before returning
    SceneDetectionsContract.model_validate(contract)
    return contract, display_img


def main():
    np.random.seed(101)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Wild Drone Inference Pipeline on device: {device}")

    # Load Dataset Test Split
    data = np.load("data/processed/ism.npz", allow_pickle=True)
    X_te, y_te, classes = data["X"][data["te"]], data["y"][data["te"]], data["classes"]

    # Load U-Net and ThermalCNN
    unet_ckpt = torch.load("models/drone_unet.pt", map_location=device, weights_only=False)
    model_unet = PanelUNet().to(device)
    model_unet.load_state_dict(unet_ckpt["state_dict"])
    model_unet.eval()

    cnn_ckpt = torch.load("models/thermal_cnn.pt", map_location=device, weights_only=False)
    model_cnn = ThermalCNN(n_classes=len(classes)).to(device)
    model_cnn.load_state_dict(cnn_ckpt["state_dict"])
    model_cnn.eval()

    with open("contracts/class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    out_dir = Path("demo/scenes")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate 2 Wild Rotated Drone Flights over Hot Desert Sand
    drone_scenarios = [
        ("drone_flight_angled_01", 3, 5, 14.5, True),
        ("drone_flight_sandstorm_02", 3, 6, -16.0, False),
    ]

    for scene_id, rows, cols, angle, force_crit in drone_scenarios:
        v_img, t_map, _, gt_panels = generate_wild_drone_frame(
            X_te, y_te, classes, rows=rows, cols=cols, angle_deg=angle, force_critical=force_crit
        )

        contract, annotated_png = analyze_drone_frame(
            v_img, t_map, scene_id, "radiometric",
            model_unet, model_cnn, cnn_ckpt, class_mapping, device
        )

        np.savez_compressed(out_dir / f"{scene_id}.npz", img=v_img, temp=t_map)
        with open(out_dir / f"{scene_id}.json", "w") as f:
            json.dump(contract, f, indent=2)
        with open(out_dir / f"{scene_id}_gt.json", "w") as f:
            json.dump(gt_panels, f, indent=2)
        cv2.imwrite(str(out_dir / f"{scene_id}.png"), annotated_png)

        print(
            f" -> Saved {scene_id} (angle={angle}°): Segmented {contract['summary']['n_modules']}/{len(gt_panels)} panels | "
            f"Baseline={contract['scene_baseline_temp_c']}°C | Flagged={len(contract['detections'])} actionable panels"
        )


if __name__ == "__main__":
    main()