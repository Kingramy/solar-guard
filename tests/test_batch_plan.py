# tests/test_batch_plan.py
from agent.loss_model import load_config
from agent.batch_plan import evaluate_scene_batch

def test_single_offline_module_has_negative_roi():
    config = load_config()
    single = [{"agent_category": "offline_module", "max_temp_c": 72.0, "delta_t_c": 32.0}]
    res = evaluate_scene_batch(single, config)
    assert res["dispatch_recommended"] is False
    assert abs(res["financials"]["batch_net_benefit_usd"] - (-39.32)) <= 0.05

def test_six_offline_modules_share_one_truck_roll():
    config = load_config()
    batch_six = [
        {"agent_category": "offline_module", "max_temp_c": 72.0, "delta_t_c": 32.0}
        for _ in range(6)
    ]
    res = evaluate_scene_batch(batch_six, config)
    assert res["dispatch_recommended"] is True
    assert res["total_modules_in_batch"] == 6
    assert res["financials"]["shared_truck_cost"] == 100.0
    assert res["financials"]["allocated_truck_cost_per_module"] == 16.67
    assert abs(res["financials"]["batch_net_benefit_usd"] - 264.05) <= 0.10

def test_critical_safety_module_dispatches_despite_negative_roi():
    config = load_config()
    # A 95C panel with 'none' category recovers $0 revenue, so financial ROI is -$100 (negative),
    # but dispatch_recommended MUST still be True due to safety!
    critical_only = [{"agent_category": "none", "max_temp_c": 95.0, "delta_t_c": 55.0}]
    res = evaluate_scene_batch(critical_only, config)
    assert res["critical_count"] == 1
    assert res["financials"]["batch_net_benefit_usd"] < 0
    assert res["dispatch_recommended"] is True