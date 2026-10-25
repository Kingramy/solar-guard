# tests/test_policy.py
from agent.loss_model import load_config
from agent.policy import evaluate_panel_safety_and_roi
from agent.agent import run_agent_loop

def test_absolute_temperature_trigger_ignores_healthy_class_and_null_delta():
    config = load_config()
    det = {
        "agent_category": "none",
        "max_temp_c": 95.0,
        "delta_t_c": None
    }
    assert evaluate_panel_safety_and_roi(det, config) == "CRITICAL_REVIEW_NOW"

def test_relative_delta_temperature_trigger_is_distinct_from_90c_ceiling():
    config = load_config()
    # 81.0C is below the 90.0C ceiling, but delta_t_c = 41.0C exceeds the 40.0C relative trigger
    det = {
        "agent_category": "none",
        "max_temp_c": 81.0,
        "delta_t_c": 41.0
    }
    assert evaluate_panel_safety_and_roi(det, config) == "CRITICAL_REVIEW_NOW"

def test_soiling_and_shadowing_routing():
    config = load_config()
    soiling_det = {"agent_category": "soiling", "max_temp_c": 48.0, "delta_t_c": 8.0}
    shadow_det = {"agent_category": "shadowing", "max_temp_c": 47.0, "delta_t_c": 7.0}
    assert evaluate_panel_safety_and_roi(soiling_det, config) == "ASSIGN_NEXT_CLEANING_CYCLE"
    assert evaluate_panel_safety_and_roi(shadow_det, config) == "MONITOR_ONLY"

def test_guardrail_overrides_conflicting_llm_proposal():
    config = load_config()
    # D-0002 in scene_critical_01 has a MockLLM fixture that proposes MONITOR_ONLY on a 95C panel
    det = {
        "id": "D-0002",
        "ml_class": "No-Anomaly",
        "agent_category": "none",
        "max_temp_c": 95.0,
        "delta_t_c": 54.8,
        "flag_reason": "temperature"
    }
    trace = run_agent_loop(det, "scene_critical_01", config)
    events = [step["event"] for step in trace]
    assert "guardrail_override" in events
    assert trace[-1]["event"] == "final"
    assert trace[-1]["content"] == "CRITICAL_REVIEW_NOW"