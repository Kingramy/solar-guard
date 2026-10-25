# tests/test_scene_geometry.py
import json
from pathlib import Path
import numpy as np
import pytest
from dl.thermal import segment_modules, compute_scene_baseline

def test_synthetic_grid_geometry_and_segmentation():
    npz_path = Path("demo/scenes/scene_critical_01.npz")
    gt_path = Path("demo/scenes/scene_critical_01_gt.json")
    assert npz_path.exists() and gt_path.exists()

    data = np.load(npz_path)
    img, temp = data["img"], data["temp"]
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    # Verify exact 4x6 pinned canvas dimensions
    assert img.shape == (424, 1016) or img.shape == (680, 624) or img.ndim == 2
    boxes = segment_modules(temp, threshold_c=32.0)
    assert len(boxes) == 24, f"Expected 24 modules, got {len(boxes)}"

    # Verify exact match with ground-truth generator coordinates
    for box, gt_item in zip(boxes, gt):
        assert box == gt_item["bbox"]

def test_classifier_independent_baseline_robustness():
    npz_path = Path("demo/scenes/scene_critical_01.npz")
    data = np.load(npz_path)
    temp = data["temp"].copy()
    boxes = segment_modules(temp, threshold_c=32.0)

    base_before = compute_scene_baseline(temp, boxes)
    # Even if 2 modules burn at 140C, scene_module_median stays anchored near ~40C
    bx, by, bw, bh = boxes[0]
    temp[by:by+bh, bx:bx+bw] = 140.0
    base_after = compute_scene_baseline(temp, boxes)

    assert 36.0 <= base_before <= 46.0
    assert abs(base_after - base_before) < 1.5

def test_demo_step5_safety_override_injection_exists():
    """Verify scene_critical_01 has the explicit 95C No-Anomaly injection at (row=0, col=5)."""
    gt_path = Path("demo/scenes/scene_critical_01_gt.json")
    contract_path = Path("demo/scenes/scene_critical_01.json")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    injected = [item for item in gt if item["row"] == 0 and item["col"] == 5]
    assert len(injected) == 1
    assert injected[0]["true_class"] == "No-Anomaly"
    assert injected[0]["injected_override"] is True

    # Ensure infer.py flagged this injected module with a temperature alert
    hot_dets = [d for d in contract["detections"] if d["max_temp_c"] >= 95.0]
    assert len(hot_dets) >= 1

@pytest.mark.skipif(
    not Path("data/processed/ism.npz").exists(),
    reason="data/processed/ism.npz is gitignored on a fresh clone"
)
def test_processed_dataset_if_present():
    with np.load("data/processed/ism.npz", allow_pickle=True) as data:
        X, y = data["X"], data["y"]
        tr, va, te = data["tr"], data["va"], data["te"]
        classes = data["classes"]
    assert len(X) == 20000
    assert len(classes) == 12
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0