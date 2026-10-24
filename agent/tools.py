# agent/tools.py
from typing import Dict, Any
try:
    from .policy import evaluate_panel_safety_and_roi
except (ImportError, ModuleNotFoundError):
    from policy import evaluate_panel_safety_and_roi

def get_solar_irradiance(sector: str, forecast_days: int) -> Dict[str, Any]:
    return {"source": "simulated", "peak_sun_hours": 6.0, "dust_risk": "low"}

def get_dispatch_cost(sector: str, n_panels: int) -> Dict[str, Any]:
    return {"one_site_visit_cost_usd": 100.0, "note": "Shared across all selected panels"}

def check_safety(max_temp_c: float, delta_t_c: float, category: str, config: Dict[str, Any]) -> Dict[str, Any]:
    det = {"agent_category": category, "max_temp_c": max_temp_c, "delta_t_c": delta_t_c}
    decision = evaluate_panel_safety_and_roi(det, config)
    return {"policy_decision": decision, "source": "deterministic_guardrail"}

def get_maintenance_schedule(sector: str) -> Dict[str, Any]:
    return {"days_until_next_cleaning": 18, "source": "simulated"}

TOOL_REGISTRY = {
    "get_solar_irradiance": get_solar_irradiance,
    "get_dispatch_cost": get_dispatch_cost,
    "check_safety": check_safety,
    "get_maintenance_schedule": get_maintenance_schedule
}