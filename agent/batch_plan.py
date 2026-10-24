# agent/batch_plan.py
from typing import Dict, Any, List
try:
    from .loss_model import calculate_revenue_loss, calculate_repair_cost
    from .policy import evaluate_panel_safety_and_roi
except (ImportError, ModuleNotFoundError):
    from loss_model import calculate_revenue_loss, calculate_repair_cost
    from policy import evaluate_panel_safety_and_roi

def evaluate_scene_batch(detections: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates all detections in a scene, filters out non-actionable items, 
    and computes the shared-visit economics.
    """
    critical_items = []
    repair_items = []
    
    total_rev_365 = 0.0
    total_comp_cost = 0.0
    annual_days = config["financial"]["annual_days"]
    
    # 1. Triage all detections through the deterministic policy
    for det in detections:
        decision = evaluate_panel_safety_and_roi(det, config)
        
        if decision == "CRITICAL_REVIEW_NOW":
            critical_items.append(det)
        elif decision in ["SCHEDULE_REPAIR", "BUNDLE_FOR_BATCH_VISIT"]:
            repair_items.append(det)
        else:
            continue # Skip MONITOR_ONLY and ASSIGN_NEXT_CLEANING_CYCLE for this specific truck roll
            
        # Accumulate economics for actionable items
        category = det["agent_category"]
        total_rev_365 += calculate_revenue_loss(category, annual_days, config)
        total_comp_cost += calculate_repair_cost(category, config)

    # 2. Calculate Batch Economics
    total_items = len(critical_items) + len(repair_items)
    shared_truck_cost = float(config["financial"]["truck_roll_usd"]) if total_items > 0 else 0.0
    
    batch_net_benefit = total_rev_365 - total_comp_cost - shared_truck_cost
    
    # 3. Final Dispatch Recommendation
    # Always dispatch if there is a critical safety item, even if ROI is negative.
    # Otherwise, dispatch only if the shared batch turns a profit.
    dispatch_recommended = (len(critical_items) > 0) or (batch_net_benefit > 0)
    
    allocated_truck = round(shared_truck_cost / total_items, 2) if total_items > 0 else 0.0
    
    return {
        "dispatch_recommended": dispatch_recommended,
        "total_modules_in_batch": total_items,
        "critical_count": len(critical_items),
        "repair_count": len(repair_items),
        "financials": {
            "total_revenue_recovered_365": round(total_rev_365, 2),
            "total_component_cost": round(total_comp_cost, 2),
            "shared_truck_cost": round(shared_truck_cost, 2),
            "batch_net_benefit_usd": round(batch_net_benefit, 2),
            "allocated_truck_cost_per_module": allocated_truck
        }
    }

if __name__ == "__main__":
    try:
        from .loss_model import load_config
    except (ImportError, ModuleNotFoundError):
        from loss_model import load_config
    config = load_config()
    
    # Simulate a batch of 6 Offline Modules
    # 1 offline module standalone = -$39.32 ROI. 6 batched = +$264.05 ROI.
    mock_batch = [
        {"agent_category": "offline_module", "max_temp_c": 75.0, "delta_t_c": 35.0} 
        for _ in range(6)
    ]
    
    result = evaluate_scene_batch(mock_batch, config)
    
    print("--- Batch Planner Results for 6 Offline Modules ---")
    print(f"Dispatch Recommended: {result['dispatch_recommended']}")
    print(f"Total Revenue Recovered: ${result['financials']['total_revenue_recovered_365']}")
    print(f"Total Component Cost: ${result['financials']['total_component_cost']}")
    print(f"Shared Truck Cost: ${result['financials']['shared_truck_cost']}")
    print(f"Net Batch Benefit: ${result['financials']['batch_net_benefit_usd']}")
    print(f"Allocated Truck Cost/Module: ${result['financials']['allocated_truck_cost_per_module']}")