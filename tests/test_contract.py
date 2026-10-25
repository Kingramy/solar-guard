# tests/test_contract.py
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from contracts.schema import SceneDetectionsContract

def test_canonical_example_validates():
    path = Path("contracts/detections_example.json")
    assert path.exists(), "contracts/detections_example.json is missing!"
    data = json.loads(path.read_text(encoding="utf-8"))
    contract = SceneDetectionsContract.model_validate(data)
    assert contract.schema_version == "1.0"
    assert contract.image_size.width == 1016
    assert contract.image_size.height == 424
    assert contract.baseline_method == "scene_module_median"

def test_all_demo_scene_contracts_validate():
    scenes_dir = Path("demo/scenes")
    scene_files = [p for p in scenes_dir.glob("*.json") if not p.name.endswith("_gt.json")]
    assert len(scene_files) >= 5, "Expected at least 5 generated scene JSON contracts."
    
    for p in scene_files:
        raw = json.loads(p.read_text(encoding="utf-8"))
        # Verify ground truth never leaks into the agent contract
        assert "ground_truth" not in raw
        for det in raw.get("detections", []):
            assert "true_class" not in det
            assert "ground_truth" not in det
        contract = SceneDetectionsContract.model_validate(raw)
        assert contract.baseline_method == "scene_module_median"

def test_contradictory_flag_reason_rejected():
    """ml_class: 'Hot-Spot' or 'Cell' with flag_reason: 'temperature' alone must fail."""
    path = Path("contracts/detections_example.json")
    bad_data = json.loads(path.read_text(encoding="utf-8"))
    bad_data["detections"][0]["ml_class"] = "Hot-Spot"
    bad_data["detections"][0]["agent_category"] = "hotspot"
    bad_data["detections"][0]["flag_reason"] = "temperature"  # Contradiction!
    
    with pytest.raises(ValidationError):
        SceneDetectionsContract.model_validate(bad_data)