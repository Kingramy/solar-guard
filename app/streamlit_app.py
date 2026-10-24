# app/streamlit_app.py
import sys
from pathlib import Path

# Add project root (solar-guard) to sys.path so 'agent' and other modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import json
import yaml
from PIL import Image

from agent.loss_model import load_config
from agent.batch_plan import evaluate_scene_batch
from agent.agent import run_agent_loop
from agent.workorder import generate_json_workorder, generate_pdf_workorder

# --- Page Configuration ---
st.set_page_config(page_title="SolarGuard Dispatch", layout="wide", page_icon="☀️")

@st.cache_data
def get_config():
    return load_config()

def load_scene_data(scene_name: str):
    json_path = PROJECT_ROOT / f"demo/scenes/{scene_name}.json"
    img_path = PROJECT_ROOT / f"demo/scenes/{scene_name}.png"
    
    with open(json_path, "r") as f:
        contract = json.load(f)
        
    img = Image.open(img_path) if img_path.exists() else None
    return contract, img

def main():
    config = get_config()
    
    st.title("☀️ SolarGuard: Autonomous Fault & Maintenance Dispatch")
    st.markdown("**Prototype output — not a real dispatch, certified diagnosis, or fire-risk determination.**")
    
    # Initialize session state for the agent trace
    if "agent_trace" not in st.session_state:
        st.session_state.agent_trace = None
        st.session_state.current_wo_det = None
        st.session_state.pdf_bytes = None
        st.session_state.json_bytes = None

    # --- Fetch Available Scenes ---
    scenes_dir = PROJECT_ROOT / "demo" / "scenes"
    available_scenes = sorted([p.stem for p in scenes_dir.glob("*.json") if "_gt" not in p.name])
    
    if not available_scenes:
        st.error("No scenes found in demo/scenes/!")
        return

    # 3-Column Layout
    col1, col2, col3 = st.columns([1.5, 1, 1.2])
    
    with col1:
        st.subheader("1. Vision & Thermal Analysis")
        selected_scene = st.selectbox("Select Solar Array Scene", available_scenes, 
                                      on_change=lambda: st.session_state.clear())
        
        contract, img = load_scene_data(selected_scene)
        
        if img:
            st.image(img, use_container_width=True, caption=f"{selected_scene} (False-Color Inferno)")
            
        st.warning("⚠️ **SYNTHETIC TEMPERATURES — NOT A RADIOMETRIC MEASUREMENT**")
        
        st.markdown(f"**Baseline Method:** {contract['baseline_method']} ({contract['scene_baseline_temp_c']}°C)")
        
        st.markdown("### Flagged Modules")
        for det in contract["detections"]:
            with st.expander(f"Module {det['id']} | CNN: {det['ml_class']} | {det['max_temp_c']}°C"):
                st.write(f"**Confidence:** {det['confidence']:.2f}")
                st.write(f"**Agent Category:** {det['agent_category']}")
                delta_str = f"+{det['delta_t_c']}°C" if det['delta_t_c'] else "N/A"
                st.write(f"**Thermal Delta (vs Median):** {delta_str}")
                st.write(f"**Flag Trigger:** `{det['flag_reason']}`")
                
                # Button to select this specific panel for the agent
                if st.button(f"Analyze {det['id']} with AI Agent", key=f"btn_{det['id']}"):
                    st.session_state.current_wo_det = det
                    st.session_state.agent_trace = None # reset trace

    with col2:
        st.subheader("2. Batch Economics")
        batch_results = evaluate_scene_batch(contract["detections"], config)
        
        with st.expander("View Financial Assumptions"):
            st.json(config["financial"])
            
        st.metric("Actionable Modules in Sector", batch_results["total_modules_in_batch"])
        st.metric("Critical Safety Flags", batch_results["critical_count"])
        
        st.markdown("### Shared-Visit Financials")
        st.write("*(365-Day Horizon Estimate)*")
        
        f = batch_results["financials"]
        st.metric("Total Revenue Recovered", f"${f['total_revenue_recovered_365']}")
        st.metric("Total Component Cost", f"-${f['total_component_cost']}")
        st.metric("Shared Truck Roll Cost", f"-${f['shared_truck_cost']} (${f['allocated_truck_cost_per_module']}/module)")
        
        # Color code the ROI
        roi_color = "normal" if f["batch_net_benefit_usd"] > 0 else "inverse"
        st.metric("Net Batch Benefit (ROI)", f"${f['batch_net_benefit_usd']}", delta=f"${f['batch_net_benefit_usd']}", delta_color=roi_color)
        
        if batch_results["dispatch_recommended"]:
            st.success("✅ **System Recommends Dispatch** (Positive ROI or Critical Safety flag present)")
        else:
            st.info("⏸️ **Monitor Only** (Negative ROI and no Critical Safety flags)")

    with col3:
        st.subheader("3. AI Agent & Guardrail")
        
        target_det = st.session_state.current_wo_det
        if not target_det:
            st.info("👈 Select a module from the Left Panel to begin agent analysis.")
        else:
            st.markdown(f"**Target:** `{target_det['id']}` ({target_det['ml_class']})")
            
            if st.button("Run ReAct Loop & Guardrail", type="primary"):
                with st.spinner("Agent thinking..."):
                    trace = run_agent_loop(target_det, contract["scene_id"], config)
                    st.session_state.agent_trace = trace
                    
                    # Generate PDFs in the background
                    out_dir = PROJECT_ROOT / "demo" / "outputs"
                    pdf_path = generate_pdf_workorder(target_det, trace, batch_results, out_dir)
                    json_path = generate_json_workorder(target_det, trace, batch_results, out_dir)
                    
                    with open(pdf_path, "rb") as pdf_file:
                        st.session_state.pdf_bytes = pdf_file.read()
                    with open(json_path, "rb") as json_file:
                        st.session_state.json_bytes = json_file.read()

            if st.session_state.agent_trace:
                st.markdown("### Execution Trace")
                for step in st.session_state.agent_trace:
                    if step["event"] == "rationale":
                        st.info(f"🧠 **Thought:** {step['content']}")
                    elif step["event"] == "action":
                        st.code(f"🔧 Action: {step['content']}")
                    elif step["event"] == "observation":
                        st.write(f"📥 *Observation:* {step['content']}")
                    elif step["event"] == "guardrail_override":
                        st.error(f"🛡️ **GUARDRAIL OVERRIDE:** {step['content']}")
                    elif step["event"] == "final":
                        st.success(f"🏁 **FINAL DECISION:** {step['content']}")
                
                st.markdown("### Deliverables")
                st.download_button(
                    label="📄 Download PDF Work Order",
                    data=st.session_state.pdf_bytes,
                    file_name=f"WorkOrder_{target_det['id']}.pdf",
                    mime="application/pdf"
                )
                st.download_button(
                    label="⚙️ Download JSON Trace",
                    data=st.session_state.json_bytes,
                    file_name=f"Trace_{target_det['id']}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()