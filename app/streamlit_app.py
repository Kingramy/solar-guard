import sys
from pathlib import Path

# Add project root to sys.path to allow imports from agent, dl, contracts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import json
import numpy as np
import cv2
import torch
from PIL import Image

from agent.loss_model import load_config
from agent.batch_plan import evaluate_scene_batch
from agent.agent import run_agent_loop
from agent.workorder import generate_json_workorder, generate_pdf_workorder
from dl.model import ThermalCNN
from dl.drone_segmenter import PanelUNet, generate_wild_drone_frame
from dl.drone_infer import analyze_drone_frame

st.set_page_config(page_title="SolarGuard Dispatch v3.0", layout="wide", page_icon="☀️")


@st.cache_data
def get_config():
    return load_config()


@st.cache_resource
def load_vision_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("contracts/class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    cnn_ckpt = torch.load("models/thermal_cnn.pt", map_location=device, weights_only=False)
    model_cnn = ThermalCNN(n_classes=len(cnn_ckpt["classes"])).to(device)
    model_cnn.load_state_dict(cnn_ckpt["state_dict"])
    model_cnn.eval()

    model_unet = None
    unet_path = Path("models/drone_unet.pt")
    if unet_path.exists():
        unet_ckpt = torch.load(unet_path, map_location=device, weights_only=False)
        model_unet = PanelUNet().to(device)
        model_unet.load_state_dict(unet_ckpt["state_dict"])
        model_unet.eval()

    return device, model_unet, model_cnn, cnn_ckpt, class_mapping


def load_scene_data(scene_name: str):
    json_path = Path(f"demo/scenes/{scene_name}.json")
    img_path = Path(f"demo/scenes/{scene_name}.png")
    with open(json_path, "r") as f:
        contract = json.load(f)
    img = Image.open(img_path) if img_path.exists() else None
    return contract, img


def main():
    config = get_config()

    st.title("☀️ SolarGuard v3.0: Autonomous Aerial Fault Detection & Dispatch")
    st.markdown("**Prototype output — not a real dispatch, certified diagnosis, or fire-risk determination.**")

    if "agent_trace" not in st.session_state:
        st.session_state.agent_trace = None
        st.session_state.current_wo_det = None
        st.session_state.pdf_bytes = None
        st.session_state.json_bytes = None
        st.session_state.live_contract = None
        st.session_state.live_img = None

    mode = st.radio(
        "Select Operation Mode:",
        ["📂 Cached Scenes & Drone Flights (Offline Ready)", "🚁 Live Drone Image Upload / Custom Flight Generator"],
        horizontal=True
    )

    contract, display_pil = None, None

    if mode.startswith("📂"):
        scenes_dir = Path("demo/scenes")
        available_scenes = sorted([p.stem for p in scenes_dir.glob("*.json") if "_gt" not in p.name])
        if not available_scenes:
            st.error("No scenes found in demo/scenes/!")
            return
        selected_scene = st.selectbox(
            "Select Solar Array or Drone Flight Scene",
            available_scenes,
            on_change=lambda: st.session_state.update({"agent_trace": None, "current_wo_det": None})
        )
        contract, display_pil = load_scene_data(selected_scene)
    else:
        device, model_unet, model_cnn, cnn_ckpt, class_mapping = load_vision_models()
        if model_unet is None:
            st.error("models/drone_unet.pt not found! Run `python -m dl.drone_segmenter` first.")
            return

        st.markdown("### 🚁 Live Aerial Thermal Feed (U-Net Segmentation + Homography + 4-View TTA CNN)")
        up_col1, up_col2 = st.columns(2)

        with up_col1:
            uploaded_file = st.file_uploader(
                "Upload Drone Frame (.png, .jpg, or radiometric .npz)",
                type=["png", "jpg", "jpeg", "npz"]
            )
            if uploaded_file is not None and st.button("🔍 Run U-Net + Homography on Uploaded File"):
                if uploaded_file.name.endswith(".npz"):
                    npz_data = np.load(uploaded_file)
                    v_img, t_map = npz_data["img"], npz_data["temp"]
                    t_src = "radiometric"
                else:
                    pil_in = Image.open(uploaded_file).convert("L")
                    v_img = np.array(pil_in, dtype=np.uint8)
                    t_map = None
                    t_src = "estimated_visual"

                c_out, img_bgr = analyze_drone_frame(
                    v_img, t_map, "live_uploaded_drone", t_src,
                    model_unet, model_cnn, cnn_ckpt, class_mapping, device
                )
                st.session_state.live_contract = c_out
                st.session_state.live_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                st.session_state.current_wo_det = None
                st.session_state.agent_trace = None

        with up_col2:
            yaw_angle = st.slider("Simulate Drone Yaw Angle (degrees)", -20.0, 20.0, 12.5, 0.5)
            inject_crit = st.checkbox("Inject 95.5°C Stealth Thermal Spike on Healthy Module", value=True)
            if st.button("⚡ Synthesize & Segment Live Rotated Drone Flight"):
                data_npz = np.load("data/processed/ism.npz", allow_pickle=True)
                X_te, y_te, classes = data_npz["X"][data_npz["te"]], data_npz["y"][data_npz["te"]], data_npz["classes"]
                v_img, t_map, _, _ = generate_wild_drone_frame(
                    X_te, y_te, classes, rows=3, cols=5, angle_deg=yaw_angle, force_critical=inject_crit
                )
                c_out, img_bgr = analyze_drone_frame(
                    v_img, t_map, f"live_drone_{int(yaw_angle)}deg", "radiometric",
                    model_unet, model_cnn, cnn_ckpt, class_mapping, device
                )
                st.session_state.live_contract = c_out
                st.session_state.live_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                st.session_state.current_wo_det = None
                st.session_state.agent_trace = None

        contract = st.session_state.live_contract
        display_pil = st.session_state.live_img
        if contract is None:
            st.info("👆 Upload a drone thermal frame or click **Synthesize & Segment Live Rotated Drone Flight** above.")
            return

    # --- 3-Column Layout ---
    col1, col2, col3 = st.columns([1.5, 1, 1.2])

    with col1:
        st.subheader("1. Vision & Thermal Analysis")
        if display_pil:
            st.image(display_pil, use_container_width=True, caption=f"Scene: {contract['scene_id']}")

        st.warning(f"⚠️ **TEMPERATURE SOURCE: {contract['temperature_source'].upper()} — PROTOTYPE SIMULATION**")
        st.markdown(
            f"**Segmented Panels:** `{contract['summary']['n_modules']}` | "
            f"**Baseline (`{contract['baseline_method']}`):** `{contract['scene_baseline_temp_c']}°C`"
        )

        with st.expander("📊 Held-Out Test Split Evaluation Metrics"):
            metrics_path = Path("reports/metrics.json")
            cm_path = Path("reports/confusion_matrix.png")
            if metrics_path.exists():
                st.json(json.loads(metrics_path.read_text()))
            if cm_path.exists():
                st.image(str(cm_path), caption="12-Class Held-Out Confusion Matrix")

        st.markdown(f"### Flagged Actionable Modules ({len(contract['detections'])})")
        for det in contract["detections"]:
            with st.expander(
                f"Module {det['id']} (R{det['row']}C{det['col']}) | CNN: {det['ml_class']} | {det['max_temp_c']}°C"
            ):
                st.write(f"**Confidence:** {det['confidence']:.2f}")
                st.write(f"**Agent Category:** `{det['agent_category']}`")
                delta_str = f"+{det['delta_t_c']}°C" if det["delta_t_c"] is not None else "N/A"
                st.write(f"**Thermal Delta (vs Scene Median):** {delta_str}")
                st.write(f"**Flag Trigger:** `{det['flag_reason']}`")

                if st.button(f"Analyze {det['id']} with AI Agent", key=f"btn_{contract['scene_id']}_{det['id']}"):
                    st.session_state.current_wo_det = det
                    st.session_state.agent_trace = None

    with col2:
        st.subheader("2. Batch Economics")
        batch_results = evaluate_scene_batch(contract["detections"], config)

        with st.expander("View Financial Assumptions"):
            st.json(config["financial"])

        st.metric("Actionable Modules in Sector", batch_results["total_modules_in_batch"])
        st.metric("Critical Safety Flags", batch_results["critical_count"])

        st.markdown("### Shared-Visit Financials (365-Day)")
        f = batch_results["financials"]
        st.metric("Total Revenue Recovered", f"${f['total_revenue_recovered_365']}")
        st.metric("Total Component Cost", f"-${f['total_component_cost']}")
        st.metric("Shared Truck Roll Cost", f"-${f['shared_truck_cost']} (${f['allocated_truck_cost_per_module']}/module)")

        roi_color = "normal" if f["batch_net_benefit_usd"] > 0 else "inverse"
        st.metric(
            "Net Batch Benefit (ROI)",
            f"${f['batch_net_benefit_usd']}",
            delta=f"${f['batch_net_benefit_usd']}",
            delta_color=roi_color
        )

        if batch_results["dispatch_recommended"]:
            st.success("✅ **System Recommends Dispatch** (Positive ROI or Critical Safety flag present)")
        else:
            st.info("⏸️ **Monitor Only** (Negative ROI and no Critical Safety flags)")

    with col3:
        st.subheader("3. AI Agent & Guardrail")
        target_det = st.session_state.current_wo_det
        if not target_det:
            st.info("👈 Select a flagged module from the Left Panel to begin agent analysis.")
        else:
            st.markdown(f"**Target:** `{target_det['id']}` (Row {target_det['row']}, Col {target_det['col']} — `{target_det['ml_class']}`)")

            if st.button("Run ReAct Loop & Guardrail", type="primary"):
                with st.spinner("Agent thinking..."):
                    trace = run_agent_loop(target_det, contract["scene_id"], config)
                    st.session_state.agent_trace = trace

                    out_dir = Path("demo/outputs")
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