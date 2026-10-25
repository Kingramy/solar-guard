import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import torch

from contracts.schema import SceneDetectionsContract
from dl.model import ThermalCNN
from dl.radiometric import order_quad_corners
from dl.drone_segmenter import PanelUNet, CANVAS_H, CANVAS_W, CORE_SHRINK_ALPHA
from dl.drone_infer import analyze_drone_frame


def build_custom_drone_frame(
    X_crops: np.ndarray,
    y_crops: np.ndarray,
    classes: list,
    mod_assignments: Dict[tuple, Dict[str, Any]],
    angle_deg: float = 12.0,
    rows: int = 3,
    cols: int = 5,
    seed: int = 42
):
    """
    Synthesizes a realistic aerial thermal drone frame matching PanelUNet geometry.
    mod_assignments: {(r, c): {"class": "Hot-Spot", "spike": 96.5, "temp_range": (85, 105)}}
    """
    np.random.seed(seed)

    # Hot desert ground with micro-dunes and rocks
    low_res_sand = np.random.uniform(37.0, 52.0, (16, 24)).astype(np.float32)
    temp_map = cv2.resize(low_res_sand, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_CUBIC)
    temp_map += np.random.normal(0, 0.7, (CANVAS_H, CANVAS_W)).astype(np.float32)

    visual_img = np.clip((temp_map - 30.0) * 4.5, 20, 145).astype(np.uint8)

    # Desert terrain rocks/dunes
    for _ in range(np.random.randint(5, 10)):
        rx, ry = np.random.randint(10, CANVAS_W - 10), np.random.randint(10, CANVAS_H - 10)
        rad = np.random.randint(3, 7)
        cv2.circle(visual_img, (rx, ry), rad, int(np.random.randint(110, 170)), -1)
        cv2.circle(temp_map, (rx, ry), rad, float(np.random.uniform(46.0, 55.0)), -1)

    core_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    gt_panels = []

    mod_w, mod_h = 36, 54
    gap_x, gap_y = 8, 12
    total_w = cols * mod_w + (cols - 1) * gap_x
    total_h = rows * mod_h + (rows - 1) * gap_y
    start_x = (CANVAS_W - total_w) / 2.0
    start_y = (CANVAS_H - total_h) / 2.0

    cx_canvas, cy_canvas = CANVAS_W / 2.0, CANVAS_H / 2.0
    M_rot = cv2.getRotationMatrix2D((cx_canvas, cy_canvas), angle_deg, 1.0)

    default_temp_ranges = {
        "No-Anomaly": (39.0, 44.0),
        "Soiling": (47.0, 54.0),
        "Offline-Module": (64.0, 76.0),
        "Diode": (65.0, 78.0),
        "Cell": (75.0, 92.0),
        "Hot-Spot": (85.0, 105.0),
        "Cracking": (60.0, 72.0),
        "Vegetation": (36.0, 42.0),
    }

    healthy_idx = list(classes).index("No-Anomaly")

    for r in range(rows):
        for c in range(cols):
            x0 = start_x + c * (mod_w + gap_x)
            y0 = start_y + r * (mod_h + gap_y)

            local_quad = np.array([
                [x0, y0],
                [x0 + mod_w, y0],
                [x0 + mod_w, y0 + mod_h],
                [x0, y0 + mod_h]
            ], dtype=np.float32)

            ones = np.ones((4, 1), dtype=np.float32)
            pts_homo = np.hstack([local_quad, ones])
            rot_quad = order_quad_corners((M_rot @ pts_homo.T).T.astype(np.float32))
            centroid = np.mean(rot_quad, axis=0, keepdims=True)

            # Outer aluminum mounting frame
            outer_frame_quad = centroid + 1.08 * (rot_quad - centroid)
            cv2.fillConvexPoly(visual_img, outer_frame_quad.astype(np.int32), 25)

            # Panel assignment
            assignment = mod_assignments.get((r, c), {"class": "No-Anomaly", "spike": None})
            target_cls = assignment.get("class", "No-Anomaly")
            override_spike = assignment.get("spike", None)
            t_range = assignment.get("temp_range", default_temp_ranges.get(target_cls, (40.0, 45.0)))

            cls_id = list(classes).index(target_cls) if target_cls in classes else healthy_idx
            matching = np.where(y_crops == cls_id)[0]
            crop_idx = int(np.random.choice(matching))
            crop = X_crops[crop_idx]

            # Warp real crop into rotated quad
            src_corners = np.array([[0, 0], [23, 0], [23, 39], [0, 39]], dtype=np.float32)
            H_mat = cv2.getPerspectiveTransform(src_corners, rot_quad)
            cv2.warpPerspective(
                crop, H_mat, (CANVAS_W, CANVAS_H),
                dst=visual_img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
            )

            # Synthesize radiometric temperature
            base_t = float(np.random.uniform(t_range[0], t_range[1]))
            crop_temp = (crop.astype(np.float32) / 255.0) * 10.0 + (base_t - 5.0)
            if override_spike is not None:
                crop_temp[14:26, 6:18] = float(override_spike)

            cv2.warpPerspective(
                crop_temp, H_mat, (CANVAS_W, CANVAS_H),
                dst=temp_map, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
            )

            # Shrunk core mask for U-Net
            shrunk_quad = centroid + CORE_SHRINK_ALPHA * (rot_quad - centroid)
            cv2.fillConvexPoly(core_mask, shrunk_quad.astype(np.int32), 1.0)

            gt_panels.append({
                "row": r,
                "col": c,
                "polygon": rot_quad.astype(int).tolist(),
                "true_class": target_cls,
                "injected_override": override_spike is not None,
                "nominal_temp_range": t_range,
            })

    return visual_img, temp_map, core_mask, gt_panels


def main():
    print("Generating 5 Perfect Thermal Aerial Drone Test Flights for SolarGuard...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("contracts/class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    cnn_ckpt = torch.load("models/thermal_cnn.pt", map_location=device, weights_only=False)
    model_cnn = ThermalCNN(n_classes=len(cnn_ckpt["classes"])).to(device)
    model_cnn.load_state_dict(cnn_ckpt["state_dict"])
    model_cnn.eval()

    unet_ckpt = torch.load("models/drone_unet.pt", map_location=device, weights_only=False)
    model_unet = PanelUNet().to(device)
    model_unet.load_state_dict(unet_ckpt["state_dict"])
    model_unet.eval()

    data = np.load("data/processed/ism.npz", allow_pickle=True)
    X_te, y_te, classes = data["X"][data["te"]], data["y"][data["te"]], list(data["classes"])

    out_dir = Path("demo/test_uploads")
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {
            "id": "flight_01_severe_hotspot_emergency",
            "title": "Flight 01: Severe Thermal Hotspot Runaway (>95°C Safety Override)",
            "angle": 12.0,
            "seed": 101,
            "assignments": {
                (1, 2): {"class": "Hot-Spot", "spike": 96.5, "temp_range": (88.0, 98.0)},
                (2, 1): {"class": "Diode", "spike": None, "temp_range": (66.0, 75.0)},
            },
            "description": "Tests deterministic safety override. Center panel has critical 96.5°C hotspot triggering immediate emergency truck roll.",
        },
        {
            "id": "flight_02_offline_string_outage",
            "title": "Flight 02: Inverter String Outage (4 Offline Modules)",
            "angle": -15.0,
            "seed": 202,
            "assignments": {
                (2, 0): {"class": "Offline-Module", "spike": None, "temp_range": (66.0, 74.0)},
                (2, 1): {"class": "Offline-Module", "spike": None, "temp_range": (67.0, 75.0)},
                (2, 2): {"class": "Offline-Module", "spike": None, "temp_range": (68.0, 76.0)},
                (2, 3): {"class": "Offline-Module", "spike": None, "temp_range": (66.0, 73.0)},
            },
            "description": "Tests ROI economic loss model. Entire string of 4 modules is disconnected (offline), yielding large economic losses justifying dispatch.",
        },
        {
            "id": "flight_03_soiling_dust_accumulation",
            "title": "Flight 03: Post-Sandstorm Dust & Soiling Layer",
            "angle": 4.5,
            "seed": 303,
            "assignments": {
                (0, 1): {"class": "Soiling", "spike": None, "temp_range": (48.0, 53.0)},
                (1, 3): {"class": "Soiling", "spike": None, "temp_range": (49.0, 54.0)},
                (2, 4): {"class": "Soiling", "spike": None, "temp_range": (48.0, 52.0)},
            },
            "description": "Tests economic cleaning deferral. Mild temperature delta (8-12°C above baseline) from desert dust layer. Non-emergency batching.",
        },
        {
            "id": "flight_04_mixed_anomalies_angled",
            "title": "Flight 04: High-Angle Multi-Fault Survey (Cell, Diode, Cracking)",
            "angle": 18.0,
            "seed": 404,
            "assignments": {
                (0, 3): {"class": "Cell", "spike": None, "temp_range": (78.0, 91.0)},
                (1, 1): {"class": "Diode", "spike": None, "temp_range": (68.0, 77.0)},
                (1, 4): {"class": "Cracking", "spike": None, "temp_range": (62.0, 71.0)},
                (2, 2): {"class": "Hot-Spot", "spike": 88.0, "temp_range": (82.0, 92.0)},
            },
            "description": "Tests multi-class vision classifier under steep 18° yaw angle with diverse fault categories and homography rectification.",
        },
        {
            "id": "flight_05_nominal_healthy_array",
            "title": "Flight 05: Nominal Clean Commercial Array (Negative Control)",
            "angle": -8.5,
            "seed": 505,
            "assignments": {},  # All modules No-Anomaly
            "description": "Negative control test. Uniform baseline operating temperatures (39-42°C), zero hotspots, zero false alarms, no dispatch recommended.",
        },
    ]

    summary_records = []

    for sc in scenarios:
        sid = sc["id"]
        v_img, t_map, core_mask, gt_panels = build_custom_drone_frame(
            X_te, y_te, classes,
            mod_assignments=sc["assignments"],
            angle_deg=sc["angle"],
            rows=3, cols=5,
            seed=sc["seed"]
        )

        # 1. Run pipeline analysis to verify and generate validation contract
        contract, annotated_bgr = analyze_drone_frame(
            v_img, t_map, sid, "radiometric",
            model_unet, model_cnn, cnn_ckpt, class_mapping, device
        )

        # 2. Save NPZ (Radiometric)
        npz_path = out_dir / f"{sid}.npz"
        np.savez_compressed(npz_path, img=v_img, temp=t_map)

        # 3. Save standard PNG (Grayscale thermal feed for direct drag-and-drop upload)
        png_path = out_dir / f"{sid}.png"
        cv2.imwrite(str(png_path), v_img)

        # 4. Save standard JPG (for users who prefer .jpg upload)
        jpg_path = out_dir / f"{sid}.jpg"
        cv2.imwrite(str(jpg_path), v_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 5. Save False-Color Inferno preview (so user can visually inspect temperature heatmaps)
        inferno_vis = cv2.applyColorMap(v_img, cv2.COLORMAP_INFERNO)
        inferno_path = preview_dir / f"{sid}_inferno.png"
        cv2.imwrite(str(inferno_path), inferno_vis)

        # 6. Save annotated detection result preview
        annotated_path = preview_dir / f"{sid}_detected_annotated.png"
        cv2.imwrite(str(annotated_path), annotated_bgr)

        # 7. Save ground truth and contract JSON
        with open(out_dir / f"{sid}_contract.json", "w") as f:
            json.dump(contract, f, indent=2)

        summary_records.append({
            "id": sid,
            "title": sc["title"],
            "angle": sc["angle"],
            "files": {
                "png": str(png_path),
                "jpg": str(jpg_path),
                "npz": str(npz_path),
                "inferno_preview": str(inferno_path),
            },
            "detected_modules": contract["summary"]["n_modules"],
            "flagged_anomalies": len(contract["detections"]),
            "baseline_c": contract["scene_baseline_temp_c"],
            "description": sc["description"],
        })

        print(f"[*] Generated {sid}: {contract['summary']['n_modules']}/15 panels segmented | Flagged: {len(contract['detections'])} | Baseline: {contract['scene_baseline_temp_c']} C")

    # Generate test upload guide markdown
    readme_content = "# SolarGuard: 5 Drone Thermal Test Flights\n\n"
    readme_content += "These 5 thermal frames are prepared for testing the Live Drone Image Upload / Custom Flight Generator tab in SolarGuard.\n\n"
    readme_content += "| Flight ID | Scenario | Angle | Segmented Panels | Flagged Issues | Formats |\n"
    readme_content += "|---|---|---|---|---|---|\n"
    for r in summary_records:
        readme_content += f"| `{r['id']}` | **{r['title']}** | {r['angle']} deg | {r['detected_modules']}/15 | {r['flagged_anomalies']} | `.png`, `.jpg`, `.npz` |\n"

    readme_content += "\n## How to Test in the Dashboard:\n"
    readme_content += "1. Open the SolarGuard dashboard at `http://localhost:8501`\n"
    readme_content += "2. Select **Live Drone Image Upload / Custom Flight Generator**\n"
    readme_content += "3. Click **Browse files** and pick any `.png`, `.jpg`, or `.npz` from `demo/test_uploads/`\n"
    readme_content += "4. Click **Run U-Net + Homography on Uploaded File**\n"
    readme_content += "5. Observe the segmented panel grid, thermal baseline, flagged modules, and run the agent dispatch loop!\n"

    (out_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"\nAll 5 test flights generated and saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
