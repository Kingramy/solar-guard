# agent/loss_model.py
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    config_path = Path("config.yaml")
    if not config_path.exists():
        fallback = Path(__file__).resolve().parent.parent / "config.yaml"
        if fallback.exists():
            config_path = fallback
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def calculate_revenue_loss(category: str, days: int, config: Dict[str, Any]) -> float:
    """
    E_lost = P_kw * H_sun * D_days * f_loss
    R_lost = E_lost * C_energy
    """
    fin = config["financial"]
    kw = fin["rated_module_kw"]
    sun = fin["peak_sun_hours_per_day"]
    price = fin["energy_value_usd_per_kwh"]
    
    # Get the specific loss fraction for this fault (e.g., hotspot = 0.10, offline = 1.0)
    f_loss = fin["production_loss_fraction"].get(category, 0.0)
    
    e_lost = kw * sun * days * f_loss
    r_lost = e_lost * price
    
    return round(r_lost, 2)

def calculate_repair_cost(category: str, config: Dict[str, Any]) -> float:
    return float(config["financial"]["repair_cost_usd"].get(category, 0.0))

def calculate_single_visit_net(revenue_recovered: float, repair_cost: float, config: Dict[str, Any]) -> float:
    """
    Net = Revenue - Component Cost - $100 Truck Roll
    """
    truck_cost = float(config["financial"]["truck_roll_usd"])
    return round(revenue_recovered - repair_cost - truck_cost, 2)