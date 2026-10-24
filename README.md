# ☀️ SolarGuard: Autonomous Fault Detection & Maintenance Dispatch

SolarGuard is an autonomous AI-driven photovoltaic (PV) fault detection and maintenance dispatch system. It combines deep learning computer vision on thermal imaging, economic loss modeling, a ReAct agent loop with deterministic safety guardrails, and automated work-order generation (PDF & JSON).

---

## 🚀 Key Features

- **Thermal Computer Vision**: PyTorch-based CNN (`ThermalCNN`) classifying PV panel anomalies (Hotspots, Offline strings, Cell defects, Soiling/Dust, etc.).
- **Economic Loss & Batch Planning**: Calculates revenue losses based on solar irradiance, rated kW, and utility energy prices to recommend ROI-positive truck dispatches.
- **AI Agent with Deterministic Guardrails**: A bounded ReAct loop with policy guardrails that override any potential hallucinated actions to guarantee safety compliance.
- **Automated Work Orders**: Generates verifiable work orders in JSON and professional ReportLab PDF formats ready for field technicians.
- **Interactive Dashboard**: A Streamlit web application for visual analysis, batch economics review, and running the agent dispatch pipeline.

---

## 📁 Repository Structure

```text
solar-guard/
├── agent/            # Agent logic, tools, LLM interface, loss model & policy guardrails
├── app/              # Streamlit web dashboard (streamlit_app.py)
├── contracts/        # Pydantic schemas and interface contracts
├── demo/             # Sample false-color inferno scenes and test cases
├── dl/               # Deep learning pipeline (model architecture, training, evaluation)
├── models/           # Trained PyTorch model weights (thermal_cnn.pt)
├── reports/          # Classification reports and confusion matrices
├── config.yaml       # System configuration & financial parameters
└── requirements.txt  # Project dependencies
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.12)
- Virtual environment tool (`venv` or `conda`)

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Kingramy/solar-guard.git
cd solar-guard

# Create and activate virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 🔒 Safety & Guardrails
SolarGuard features deterministic safety guardrails: any module exhibiting severe temperature deltas (e.g. hotspot > 90°C or $\Delta T > 50\text{°C}$) automatically triggers immediate critical review, superseding any conflicting model recommendations.
