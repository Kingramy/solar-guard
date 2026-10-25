# SolarGuard: 5 Drone Thermal Test Flights

These 5 thermal frames are prepared for testing the Live Drone Image Upload / Custom Flight Generator tab in SolarGuard.

| Flight ID | Scenario | Angle | Segmented Panels | Flagged Issues | Formats |
|---|---|---|---|---|---|
| `flight_01_severe_hotspot_emergency` | **Flight 01: Severe Thermal Hotspot Runaway (>95°C Safety Override)** | 12.0 deg | 15/15 | 6 | `.png`, `.jpg`, `.npz` |
| `flight_02_offline_string_outage` | **Flight 02: Inverter String Outage (4 Offline Modules)** | -15.0 deg | 15/15 | 2 | `.png`, `.jpg`, `.npz` |
| `flight_03_soiling_dust_accumulation` | **Flight 03: Post-Sandstorm Dust & Soiling Layer** | 4.5 deg | 15/15 | 3 | `.png`, `.jpg`, `.npz` |
| `flight_04_mixed_anomalies_angled` | **Flight 04: High-Angle Multi-Fault Survey (Cell, Diode, Cracking)** | 18.0 deg | 15/15 | 8 | `.png`, `.jpg`, `.npz` |
| `flight_05_nominal_healthy_array` | **Flight 05: Nominal Clean Commercial Array (Negative Control)** | -8.5 deg | 15/15 | 3 | `.png`, `.jpg`, `.npz` |

## How to Test in the Dashboard:
1. Open the SolarGuard dashboard at `http://localhost:8501`
2. Select **Live Drone Image Upload / Custom Flight Generator**
3. Click **Browse files** and pick any `.png`, `.jpg`, or `.npz` from `demo/test_uploads/`
4. Click **Run U-Net + Homography on Uploaded File**
5. Observe the segmented panel grid, thermal baseline, flagged modules, and run the agent dispatch loop!
