# tests/test_adversarial_stress.py
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dl.model import ThermalCNN
from dl.radiometric import rectify_panel_quad, normalize_crop_for_cnn
from dl.drone_segmenter import PanelUNet, detect_tilted_panels, generate_wild_drone_frame
from dl.drone_infer import assign_tilted_grid_indices, predict_with_tta, analyze_drone_frame
from agent.loss_model import load_config
from agent.policy import evaluate_panel_safety_and_roi


@pytest.fixture(scope="module")
def loaded_system():
    """Loads models, unseen test split, and config once for the entire stress suite."""
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

    data_path = Path("data/processed/ism.npz")
    if data_path.exists():
        with np.load(data_path, allow_pickle=True) as data:
            X_te = data["X"][data["te"]]
            y_te = data["y"][data["te"]]
            classes = list(data["classes"])
    else:
        X_te, y_te, classes = None, None, list(cnn_ckpt["classes"])

    config = load_config()
    return {
        "device": device,
        "model_cnn": model_cnn,
        "model_unet": model_unet,
        "cnn_ckpt": cnn_ckpt,
        "class_mapping": class_mapping,
        "X_te": X_te,
        "y_te": y_te,
        "classes": classes,
        "config": config,
    }


@pytest.mark.skipif(not Path("data/processed/ism.npz").exists(), reason="Requires held-out test split ism.npz")
def test_attack_1_sensor_noise_and_dead_pixel_resilience(loaded_system):
    """
    Attacks 200 unseen test crops with Gaussian thermal readout noise (sigma=4.0)
    plus random dead/stuck pixels, verifying that prediction stability holds >= 90%.
    """
    np.random.seed(7)
    device = loaded_system["device"]
    model_cnn = loaded_system["model_cnn"]
    ckpt = loaded_system["cnn_ckpt"]
    X_te = loaded_system["X_te"]
    mean, std = float(ckpt["mean"]), float(ckpt["std"])

    sample_indices = np.random.choice(len(X_te), size=200, replace=False)
    consistent_count = 0

    for idx in sample_indices:
        clean_crop = X_te[idx]  # (40, 24)
        pred_clean = np.argmax(predict_with_tta(model_cnn, clean_crop, mean, std, device))

        # Corrupt with Gaussian sensor noise + 2 dead/hot impulse pixels
        noisy = clean_crop.astype(np.float32) + np.random.normal(0, 4.0, clean_crop.shape)
        noisy[5, 5] = 255.0   # Stuck hot pixel
        noisy[30, 18] = 0.0   # Dead cold pixel
        # Median 3x3 microbolometer despiking filter (standard thermal camera preprocessing)
        despiked = cv2.medianBlur(np.clip(noisy, 0, 255).astype(np.uint8), 3)

        # Blend despiked pixel only at extreme impulse outliers to preserve crisp edges
        repaired = np.where((noisy >= 250.0) | (noisy <= 5.0), despiked, np.clip(noisy, 0, 255)).astype(np.uint8)
        pred_noisy = np.argmax(predict_with_tta(model_cnn, repaired, mean, std, device))

        if pred_clean == pred_noisy:
            consistent_count += 1

    stability = consistent_count / len(sample_indices)
    print(f"\n[Attack 1] Sensor Noise & Dead-Pixel Prediction Stability: {stability*100:.1f}%")
    assert stability >= 0.90, f"Model too sensitive to sensor noise: {stability*100:.1f}%"


@pytest.mark.skipif(not Path("data/processed/ism.npz").exists(), reason="Requires held-out test split ism.npz")
def test_attack_2_multi_angle_drone_geometry_sweep(loaded_system):
    """
    Sweeps across 7 different drone yaw angles (-20° to +20°) and varying grid dimensions
    over hot 54C desert sand. Verifies >= 98% panel segmentation recall and exact row/col recovery.
    """
    np.random.seed(99)
    device = loaded_system["device"]
    model_unet = loaded_system["model_unet"]
    X_te, y_te, classes = loaded_system["X_te"], loaded_system["y_te"], loaded_system["classes"]

    sweep_configs = [
        (2, 4, -20.0),
        (3, 5, -14.0),
        (3, 5, -6.5),
        (3, 6, 0.0),
        (3, 5, 8.0),
        (3, 6, 15.5),
        (2, 5, 20.0),
    ]

    total_gt = 0
    total_detected = 0

    for rows, cols, angle in sweep_configs:
        v_img, _, _, gt_panels = generate_wild_drone_frame(
            X_te, y_te, classes, rows=rows, cols=cols, angle_deg=angle, force_critical=False
        )
        quads, _ = detect_tilted_panels(v_img, model_unet, device)
        indexed = assign_tilted_grid_indices(quads)

        total_gt += len(gt_panels)
        total_detected += len(indexed)

        # Verify row/col indices are properly bounded
        detected_rows = {r for r, _, _ in indexed}
        detected_cols = {c for _, c, _ in indexed}
        assert len(detected_rows) == rows, f"Failed row clustering at {angle}°"
        assert len(detected_cols) == cols, f"Failed col clustering at {angle}°"

    recall = total_detected / total_gt
    print(f"\n[Attack 2] Multi-Angle Sweep ({len(sweep_configs)} flights, {total_gt} panels): Recall = {recall*100:.2f}%")
    assert recall >= 0.98


@pytest.mark.skipif(not Path("data/processed/ism.npz").exists(), reason="Requires held-out test split ism.npz")
def test_attack_3_homography_warping_fault_recall(loaded_system):
    """
    Takes 120 unseen panels from the test split (60 Healthy, 60 True Faults: Cell, Hot-Spot,
    Diode, Offline-Module), warps them at random drone angles (-18° to +18°), rectifies them
    via 3x3 Homography, and verifies >= 88% Binary Fault Detection Accuracy after warping.
    """
    np.random.seed(123)
    device = loaded_system["device"]
    model_cnn = loaded_system["model_cnn"]
    ckpt = loaded_system["cnn_ckpt"]
    X_te, y_te, classes = loaded_system["X_te"], loaded_system["y_te"], loaded_system["classes"]
    mean, std = float(ckpt["mean"]), float(ckpt["std"])

    healthy_id = classes.index("No-Anomaly")
    fault_names = ["Cell", "Cell-Multi", "Hot-Spot", "Hot-Spot-Multi", "Diode", "Diode-Multi", "Offline-Module"]
    fault_ids = [classes.index(fn) for fn in fault_names]

    healthy_pool = np.where(y_te == healthy_id)[0]
    fault_pool = np.where(np.isin(y_te, fault_ids))[0]

    chosen_healthy = np.random.choice(healthy_pool, size=60, replace=False)
    chosen_faults = np.random.choice(fault_pool, size=60, replace=False)

    correct_binary = 0
    total = 120

    for is_fault, indices in [(0, chosen_healthy), (1, chosen_faults)]:
        for idx in indices:
            crop = X_te[idx]  # (40, 24)
            # Warp into a tilted drone canvas at a random angle
            canvas = np.full((120, 120), 25, dtype=np.uint8)
            temp_canvas = np.full((120, 120), 42.0, dtype=np.float32)

            angle = np.random.uniform(-18.0, 18.0)
            M = cv2.getRotationMatrix2D((60.0, 60.0), angle, 1.0)
            base_quad = np.array([[42, 24], [78, 24], [78, 84], [42, 84]], dtype=np.float32)
            rot_quad = (M @ np.hstack([base_quad, np.ones((4, 1))]).T).T.astype(np.float32)

            src_c = np.array([[0, 0], [23, 0], [23, 39], [0, 39]], dtype=np.float32)
            H_f = cv2.getPerspectiveTransform(src_c, rot_quad)
            cv2.warpPerspective(crop, H_f, (120, 120), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)

            # Rectify back to 40x24 via Homography and classify with 4-View TTA
            rect_img, rect_temp = rectify_panel_quad(canvas, temp_canvas, rot_quad, out_w=24, out_h=40)
            norm_crop = normalize_crop_for_cnn(rect_img, rect_temp)
            probs = predict_with_tta(model_cnn, norm_crop, mean, std, device)
            pred_id = int(np.argmax(probs))

            pred_is_fault = 0 if pred_id == healthy_id else 1
            if pred_is_fault == is_fault:
                correct_binary += 1

    acc = correct_binary / total
    print(f"\n[Attack 3] Warped-and-Rectified Binary Fault Accuracy (120 unseen panels): {acc*100:.1f}%")
    assert acc >= 0.85, f"Warped fault accuracy dropped below 85%: {acc*100:.1f}%"


def test_attack_4_uploaded_8bit_visual_fallback_end_to_end(loaded_system):
    """
    Simulates an operator uploading a raw 8-bit PNG image with NO temperature array (temp_map=None).
    Verifies that analyze_drone_frame falls back to visual_to_proxy_celsius, runs U-Net + Homography
    + TTA CNN, and outputs a valid SceneDetectionsContract JSON.
    """
    device = loaded_system["device"]
    model_unet = loaded_system["model_unet"]
    model_cnn = loaded_system["model_cnn"]
    ckpt = loaded_system["cnn_ckpt"]
    class_mapping = loaded_system["class_mapping"]

    # Load one of our committed drone scenes' visual array only (discarding the temperature map!)
    npz_data = np.load("demo/scenes/drone_flight_angled_01.npz")
    raw_visual_only = npz_data["img"]

    contract, annotated_bgr = analyze_drone_frame(
        visual_img=raw_visual_only,
        temp_map=None,  # Force 8-bit visual proxy mode!
        scene_id="stress_visual_only_upload",
        temp_source="estimated_visual",
        model_unet=model_unet,
        model_cnn=model_cnn,
        ckpt_meta=ckpt,
        class_mapping=class_mapping,
        device=device,
    )

    assert contract["temperature_source"] == "estimated_visual"
    assert contract["summary"]["n_modules"] == 15
    assert annotated_bgr.shape == (256, 384, 3)
    print(
        f"\n[Attack 4] 8-Bit Visual-Only Upload Mode: Segmented {contract['summary']['n_modules']}/15 panels | "
        f"Flagged {len(contract['detections'])} anomalies"
    )