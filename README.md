# Flame in Freefall

Predictive fire safety system for microgravity habitats.

- Live Mission Control App: [Launch Interactive Dashboard](https://flame-in-freefall-nasa.streamlit.app/)
- Video Briefing: [Watch Project Demo](https://www.youtube.com/watch?v=NvrEwwqwaJw)
- NASA Data Sources: NASA Physical Science Informatics (PSI) Archive - FLEX (PSI-69), BASS-II (PSI-25), and SAME (PSI-102)

## Background

On Earth, buoyancy lifts hot combustion gases upward into ceiling smoke detectors. In microgravity, natural convection disappears:

- Smoke drifts sideways with cabin ventilation instead of rising, delaying standard ceiling alarms.
- Spherical diffusion flames and low-temperature cool flames can burn with minimal visible light while generating carbon monoxide.
- Enriched-oxygen exploration atmospheres planned for Lunar Gateway, Artemis habitats, and Mars transit vehicles increase flashover risk.

Flame in Freefall is an Environmental Control and Life Support System (ECLSS) safety tool. It summarizes NASA ISS combustion experiments, ranks cabin fire hazards on a 100-point severity index, and interprets live sensor telemetry to recommend suppression actions.

## Model Performance

Trained and evaluated on 393 NASA ISS experimental runs spanning liquid droplets, solid polymers, fabrics, and wire insulation:

- Fire Risk Classifier (5-Fold CV): 89.87% accuracy across SAFE, WARNING, and CRITICAL regimes
- Critical Fire Recall (Holdout): 95.0% sensitivity to flashover and toxic smoldering hazards
- Catastrophic False Negatives: 0 out of 40 Critical fires misclassified as Safe on unseen test runs
- Extinction Time Regressor: R2 of 0.76 with 2.50s mean absolute error

## Physics-Informed Features

Instead of feeding raw cabin logs directly into a model, the preprocessing pipeline derives thermodynamic and kinetic invariants from NASA PSI telemetry:

- Oxygen Partial Pressure: Dalton's Law oxidizer availability (cabin pressure multiplied by oxygen fraction).
- Gas Mixture Thermal Conductivity: Weighted thermal diffusivity across helium, nitrogen, carbon dioxide, and oxygen, capturing how helium dilution accelerates radiative flame quenching.
- D2-Law Kinematic Residence Index: Characteristic fuel burning timescale based on initial diameter and burning rate.
- Toxic Smoldering Ratio: Ratio of carbon monoxide concentration (ppm) to oxygen percentage to flag low-visibility wire pyrolysis before open flame ignition.

## Dashboard Features

1. ECLSS Telemetry Controls: Adjust cabin pressure, oxygen percentage, helium diluent percentage, carbon monoxide concentration, and fuel geometry, or load four benchmark scenarios (Nominal ISS, 26% O2 Flashover, Invisible Wire Smoldering, and 35% Helium Quench).
2. Suppression Directives: Crew and ECLSS actions including ventilation throttling, electrical bus isolation, and inert gas injection.
3. 3D Combustion Phase Space and Spherical Flame Shell: Interactive Plotly visualizations comparing current cabin conditions against all 393 NASA ISS experiments.
4. Model Validation and Safety Audit: Embedded holdout confusion matrix (0 out of 40 missed Critical fires) and extinction time parity plot.

## Running Locally

```bash
git clone [https://github.com/Prasiddha976/flame-in-freefall.git](https://github.com/Prasiddha976/flame-in-freefall.git)
cd flame-in-freefall
pip install -r requirements.txt
streamlit run app.py
