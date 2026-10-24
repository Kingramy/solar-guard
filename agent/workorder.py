# agent/workorder.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def generate_json_workorder(detection: Dict[str, Any], trace: List[Dict[str, Any]], batch_stats: Dict[str, Any], output_dir: Path) -> Path:
    """Saves the complete state and agent trace as a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract the final decision from the trace
    final_decision = trace[-1]["content"] if trace and trace[-1]["event"] == "final" else "UNKNOWN"
    
    wo_data = {
        "work_order_id": f"WO-{detection['id']}-{datetime.now().strftime('%Y%m%d%H%M')}",
        "timestamp": datetime.now().isoformat(),
        "target_module": detection,
        "batch_context": batch_stats,
        "agent_trace": trace,
        "final_decision": final_decision
    }
    
    json_path = output_dir / f"{wo_data['work_order_id']}.json"
    with open(json_path, "w") as f:
        json.dump(wo_data, f, indent=2)
        
    return json_path

def generate_pdf_workorder(detection: Dict[str, Any], trace: List[Dict[str, Any]], batch_stats: Dict[str, Any], output_dir: Path) -> Path:
    """Generates a one-page PDF work order using ReportLab."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wo_id = f"WO-{detection['id']}-{datetime.now().strftime('%Y%m%d%H%M')}"
    pdf_path = output_dir / f"{wo_id}.pdf"
    
    final_decision = trace[-1]["content"] if trace and trace[-1]["event"] == "final" else "UNKNOWN"
    
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # 1. Header & Priority Banner
    c.setFont("Helvetica-Bold", 20)
    if "CRITICAL" in final_decision:
        c.setFillColor(colors.red)
        banner_text = "PRIORITY: CRITICAL (IMMEDIATE REVIEW REQUIRED)"
    elif "REPAIR" in final_decision or "BATCH" in final_decision:
        c.setFillColor(colors.orange)
        banner_text = f"PRIORITY: {final_decision.replace('_', ' ')}"
    else:
        c.setFillColor(colors.green)
        banner_text = f"PRIORITY: {final_decision.replace('_', ' ')}"
        
    c.drawString(50, height - 50, banner_text)
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 80, f"SolarGuard Work Order: {wo_id}")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 95, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Algiers Time)")
    
    # 2. Location & Model Diagnostics
    c.line(50, height - 110, width - 50, height - 110)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 130, "Asset & Diagnostic Information")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 150, f"Site: Demo Solar Park (fictional) | Sector 4, Array B")
    c.drawString(50, height - 165, f"Location: Row {detection['row']}, Column {detection['col']} (Fictional GPS: 32.4900N, 3.6700E)")
    
    c.drawString(50, height - 190, f"CNN Predicted Class: {detection['ml_class']} (Confidence: {detection['confidence']:.2f})")
    
    # Emphasize Synthetic Temperature
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.blue)
    delta_text = f"(Delta: +{detection['delta_t_c']}C)" if detection.get('delta_t_c') else "(Delta: N/A)"
    c.drawString(50, height - 205, f"SYNTHETIC TEMPERATURE: {detection['max_temp_c']}C {delta_text}")
    c.setFillColor(colors.black)

    # 3. Economics
    c.line(50, height - 225, width - 50, height - 225)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 245, "Financial & Batch Context (Simulated Assumptions)")
    
    c.setFont("Helvetica", 10)
    if batch_stats:
        c.drawString(50, height - 265, f"Batch Size: {batch_stats['total_modules_in_batch']} actionable modules in this sector.")
        c.drawString(50, height - 280, f"Total 365-Day Revenue at Risk: ${batch_stats['financials']['total_revenue_recovered_365']}")
        c.drawString(50, height - 295, f"Shared Truck Roll Cost: ${batch_stats['financials']['shared_truck_cost']} (Allocated: ${batch_stats['financials']['allocated_truck_cost_per_module']}/module)")
        c.drawString(50, height - 310, f"Net Batch Benefit (ROI): ${batch_stats['financials']['batch_net_benefit_usd']}")
    
    # 4. Guardrail Action
    c.line(50, height - 330, width - 50, height - 330)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 350, "Final Guardrail Decision")
    
    c.setFont("Helvetica", 10)
    y_pos = height - 370
    
    # Extract the guardrail rationale if it exists
    rationale = "Policy executed successfully."
    for step in trace:
        if step["event"] == "guardrail_override":
            rationale = step["content"]
            
    # Wrap text for rationale
    c.drawString(50, y_pos, "Rationale: " + rationale[:90])
    if len(rationale) > 90:
        c.drawString(50, y_pos - 15, rationale[90:180])
    
    # 5. Mandatory Prototype Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.gray)
    c.drawString(50, 50, "Prototype output — not a real dispatch, certified diagnosis, or fire-risk determination.")
    
    c.save()
    return pdf_path

if __name__ == "__main__":
    # Test Work Order Generation
    import os
    
    test_det = {
        "id": "D-0002", "row": 0, "col": 5, 
        "ml_class": "No-Anomaly", "confidence": 0.94,
        "max_temp_c": 95.0, "delta_t_c": 54.8
    }
    
    test_trace = [
        {"event": "rationale", "content": "The CNN says No-Anomaly. I propose MONITOR_ONLY."},
        {"event": "guardrail_override", "content": "LLM proposed MONITOR_ONLY, but deterministic policy mandates CRITICAL_REVIEW_NOW. Override applied."},
        {"event": "final", "content": "CRITICAL_REVIEW_NOW"}
    ]
    
    test_batch = {
        "total_modules_in_batch": 5,
        "financials": {
            "total_revenue_recovered_365": 903.40,
            "shared_truck_cost": 100.0,
            "allocated_truck_cost_per_module": 20.0,
            "batch_net_benefit_usd": 203.40
        }
    }
    
    out_dir = Path("demo/outputs")
    pdf_path = generate_pdf_workorder(test_det, test_trace, test_batch, out_dir)
    json_path = generate_json_workorder(test_det, test_trace, test_batch, out_dir)
    
    print(f"Generated test work orders:\n- {pdf_path}\n- {json_path}")