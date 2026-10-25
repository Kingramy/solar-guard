# tests/test_economics.py
import pytest
from agent.loss_model import (
    load_config,
    calculate_revenue_loss,
    calculate_repair_cost,
    calculate_single_visit_net
)

def test_30_day_and_365_day_offline_module_math():
    config = load_config()
    # 0.55 kW * 6 h/day * 30 days * 1.0 * $0.15 = $14.85
    rev_30 = calculate_revenue_loss("offline_module", 30, config)
    assert rev_30 == 14.85

    # 0.55 kW * 6 h/day * 365 days * 1.0 * $0.15 = $180.675 -> $180.67 or $180.68 (banker's rounding)
    rev_365 = calculate_revenue_loss("offline_module", 365, config)
    assert abs(rev_365 - 180.68) <= 0.02

    # Single module visit net: $180.67/68 - $120 repair - $100 truck = -$39.32 or -$39.33
    repair = calculate_repair_cost("offline_module", config)
    net_single = calculate_single_visit_net(rev_365, repair, config)
    assert abs(net_single - (-39.32)) <= 0.02

def test_zero_loss_category():
    config = load_config()
    assert calculate_revenue_loss("none", 365, config) == 0.0
    assert calculate_repair_cost("none", config) == 0.0

def test_negative_inputs_raise_value_error():
    config = load_config()
    with pytest.raises(ValueError):
        calculate_revenue_loss("offline_module", -10, config)
    with pytest.raises(ValueError):
        calculate_revenue_loss("invalid_category_name", 30, config)