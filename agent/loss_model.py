# agent/loss_model.py
import yaml
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def calculate_revenue_loss(category: str, days: int, config: Dict[str, Any]) -> float:
    """
    E_lost = P_kw * H_sun * D_days * f_loss
    R_lost = E_lost * C_energy
    Raises ValueError on negative inputs or invalid loss fractions.
    """
    if days < 0:
        raise ValueError(f"days cannot be negative, got {days}")
        
    fin = config["financial"]
    kw = float(fin["rated_module_kw"])
    sun = float(fin["peak_sun_hours_per_day"])
    price = float(fin["energy_value_usd_per_kwh"])
    
    if kw < 0 or sun < 0 or price < 0:
        raise ValueError("Financial power, sun hours, and energy price must be non-negative.")
        
    if category not in fin["production_loss_fraction"]:
        raise ValueError(f"Unknown fault category: {category}")
        
    f_loss = float(fin["production_loss_fraction"][category])
    if not (0.0 <= f_loss <= 1.0):
        raise ValueError(f"Loss fraction must be in [0, 1], got {f_loss}")
    
    e_lost = kw * sun * days * f_loss
    r_lost = e_lost * price
    
    return round(r_lost, 2)

def calculate_repair_cost(category: str, config: Dict[str, Any]) -> float:
    if category not in config["financial"]["repair_cost_usd"]:
        raise ValueError(f"Unknown fault category: {category}")
    return float(config["financial"]["repair_cost_usd"][category])

def calculate_single_visit_net(revenue_recovered: float, repair_cost: float, config: Dict[str, Any]) -> float:
    truck_cost = float(config["financial"]["truck_roll_usd"])
    return round(revenue_recovered - repair_cost - truck_cost, 2)