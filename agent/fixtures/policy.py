# agent/policy.py
from typing import Dict, Any
try:
    from .loss_model import calculate_revenue_loss, calculate_repair_cost
except (ImportError, ModuleNotFoundError):
    from loss_model import calculate_revenue_loss, calculate_repair_cost

def evaluate_panel_safety_and_roi(detection: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    The deterministic guardrail that overrides the LLM if safety or economics demand it.
    """
    safety = config["safety_policy"]
    category = detection["agent_category"]
    max_temp = detection["max_temp_c"]
    delta_t = detection.get("delta_t_c")
    
    # --- RULE 1: Absolute Thermal Ceiling ---
    if max_temp >= safety["max_temperature_review_c"]:
        return "CRITICAL_REVIEW_NOW"
        
    # --- RULE 2: Relative Thermal Anomaly ---
    if delta_t is not None and delta_t >= safety["delta_temperature_review_c"]:
        return "CRITICAL_REVIEW_NOW"
        
    # --- RULE 3 & 4: Categorical Overrides ---
    if category == "soiling":
        return "ASSIGN_NEXT_CLEANING_CYCLE"
    if category in ["shadowing", "none"]:
        return "MONITOR_ONLY"
        
    # --- RULE 5: Economic ROI Check (365-Day Horizon) ---
    annual_days = config["financial"]["annual_days"]
    rev_365 = calculate_revenue_loss(category, annual_days, config)
    comp_cost = calculate_repair_cost(category, config)
    truck_cost = config["financial"]["truck_roll_usd"]
    
    net_standalone = rev_365 - comp_cost - truck_cost
    net_batch = rev_365 - comp_cost
    
    if net_standalone > 0:
        return "SCHEDULE_REPAIR"
    elif net_batch > 0:
        # It's not worth rolling a truck for this alone, but worth adding to a shared trip
        return "BUNDLE_FOR_BATCH_VISIT"
    else:
        return "MONITOR_ONLY"

if __name__ == "__main__":
    # Smoke Test
    try:
        from .loss_model import load_config
    except (ImportError, ModuleNotFoundError):
        from loss_model import load_config
    config = load_config()
    
    # Simulate Demo Step 5: The LLM thinks it's healthy, but it's 95 degrees
    test_detection = {
        "agent_category": "none",
        "max_temp_c": 95.0,
        "delta_t_c": 55.0
    }
    
    decision = evaluate_panel_safety_and_roi(test_detection, config)
    print(f"Safety Guardrail Output for 95C 'Healthy' Panel: {decision}")
    assert decision == "CRITICAL_REVIEW_NOW"