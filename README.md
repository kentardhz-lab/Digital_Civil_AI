# Digital Civil AI

This repository demonstrates a Python-based data analysis and machine learning pipeline
for civil engineering applications, with a focus on structural elements, engineering-driven
feature design, and rule-based validation logic.

---

## Phase 1 – Structural Data Analysis (Completed)

Phase 1 focuses on building a clean and reproducible engineering data workflow
using structural element data.

### Implemented steps
- Load structural data from CSV
- Data quality checks and handling of missing values
- Feature engineering (e.g. load per meter)
- Visualization of engineering metrics
- Rule-based validation of unusual structural loads

---

## Phase 2 – Machine Learning Baseline (completed)

Phase 2 extends the project towards predictive modeling and machine learning.

### Current focus
- Dataset generation and feature combination
- Train / test splitting
- Baseline regression models
- Initial evaluation metrics

Outputs include:
- Baseline regression models (Linear Regression, Decision Tree)
- Evaluation metrics (MAE, RMSE, R² where applicable)
- Metrics exported to outputs/phase2_metrics.txt

This phase is under active development and will be extended with more advanced models.

---

## Project Structure

```text
Digital_Civil_AI/
├── data/
│   └── elements.csv              # Sample structural element data
│
├── data_generated/               # Generated datasets (Phase 2)
│
├── notebooks/
│   └── test.ipynb                # Exploratory analysis (Phase 1)
│
├── src/
│   └── testok.py                 # Script-based experiments
│
├── phase1-intero.py              # Phase 1 executable pipeline
├── phase2-ml-baseline.py         # Phase 2 baseline ML pipeline
├── phase2-ml-on-combined.py      # Extended ML experiments
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore

## How to Run

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Run Phase 1 
python phase1-intero.py

Run Phase 2
python phase2-ml-baseline.py

## Roadmap

- Phase 3: Feature expansion and model comparison
- Phase 4: Model validation and cross-validation
- Phase 5: Engineering-oriented interpretation of ML results
- Phase 6: Automation and AI-assisted decision suppor

## Notes

This project is designed with an engineering-first mindset.
Machine learning models are treated as supportive tools, not black-box replacements
for structural reasoning and validation.

## Phase 3 — Feature Engineering & Model Comparison

What was done:
- Engineered features: load_per_meter, inv_length
- Sanity checks for engineered features (NaN / Inf)
- Baseline vs engineered comparison using Linear Regression

### Outputs (Phase 3)
- Engineered features added: load_per_meter, inv_length
- Sanity checks: NaN/Inf validation for engineered columns
- Comparison: baseline vs engineered features (Linear Regression)
- Non-linear check: Decision Tree regressor (results unstable due to very small dataset)

### Conclusion (Phase 3)
- The dataset is very small; therefore metrics (especially R²) can fluctuate significantly.
- Feature ideas were reviewed for target leakage risk; leakage-prone features were avoided.
- This phase focused on engineering reasoning and model selection logic, not maximum performance.

Notes:
- Dataset is very small, so metrics (especially R²) are unstable.
- A leakage-prone feature idea was identified and avoided (target leakage awareness).
- Next step: try a non-linear model and/or cross-validation for small datasets.

### Non-linear model test
- Tested a simple Decision Tree regressor to capture non-linear behavior.
- Observed significant error reduction compared to linear models.
- Results are unstable due to very small dataset size.
- This step demonstrates model selection reasoning rather than final performance.

## Phase 4 — Model Validation (Cross-Validation)

What was done:
- Implemented cross-validation using KFold.
- Adjusted `n_splits` to match the extremely small dataset size (n=4).
- Compared Linear Regression vs a constrained Decision Tree (`max_depth=3`) using R².

Key takeaway:
- With very small datasets, R² can be undefined or highly unstable; variance (std) is as important as the mean.
- This phase focuses on validation discipline rather than performance claims.

Outputs:
- `outputs/phase4_cv_metrics.txt`

# Phase 5 — Engineering-Oriented Interpretation of ML Results

Documentation:
- docs/phase5-engineering-interpretation.md

Focus:
- Engineering judgment over raw ML metrics
- Interpretation of instability caused by very small datasets


## Phase 6 — Robustness & Scenario-Based Decision Support

In this phase, the project moves beyond raw ML metrics and focuses on
engineering-grade decision support under uncertainty.

### What was done
- Robustness testing via controlled noise injection on input features
- Feature sensitivity analysis (perturbation without performance metrics)
- Scenario framing using percentile-based Best / Nominal / Worst cases
- Engineering verdict layer translating ML outputs into decision ranges

### Key insights
- With extremely small datasets, performance metrics (e.g. R²) are unstable or undefined
- Robustness and behavioral consistency are more important than single-point accuracy
- Linear models provide smoother, more predictable trends
- Tree-based models highlight threshold effects and worst-case risks

### Engineering takeaway
Model outputs should be interpreted as ranges, not exact predictions.
The system is suitable for scenario comparison and risk-aware decision support,
not for deterministic numerical forecasts.

## Phase 7 — Decision Layer (Engineering Verdict)

Goal: Translate scenario-based ML outputs into an engineering-grade decision verdict.

What it does
- Builds percentile-based ranges (Best/Nominal/Worst) from the target variable
- Computes a simple, model-free sensitivity proxy (mean absolute correlation)
- Produces an Engineering Verdict: Safe / Borderline / Risky, plus confidence and recommended action

Run
```bash
python src/phase7_decision_layer.py

## Phase 8 — System Validation & Stress Testing

Goal: Validate the engineering decision system under uncertainty, degradation, and extreme conditions.

After establishing an engineering-grade decision layer in Phase 7, this phase focuses on validating the stability, consistency, and conservativeness of the system when inputs deviate from ideal conditions.

Instead of relying on classical ML performance metrics, the validation is performed from an engineering risk perspective.

### What was done
- Controlled degradation of input parameters (loads and geometrical properties)
- Simulation of extreme, worst-case engineering scenarios
- Comparison of engineering verdicts across multiple stress levels

### Validation scenarios
- Base: Original dataset without modification  
- Degraded: Increased loads and reduced effective lengths to simulate uncertainty  
- Extreme: Aggressive amplification of loads and geometry degradation  

### Output
The system generates a validation report:

Each column represents one scenario (Base / Degraded / Extreme) and contains the corresponding engineering verdict:
- Safe
- Borderline
- Risky

### Engineering insight
A reliable engineering decision system must escalate risk transparently as conditions worsen.
This phase confirms that the verdict logic behaves predictably, conservatively, and without abrupt or erratic transitions.

Run
```bash
python src/src/phase8_system_validation.py

## Phase 9 — Modular Automated Pipeline (Final Integration)

Goal: Transform the project from a collection of isolated scripts into a fully integrated, modular, and automated engineering decision system executable with a single command.

After validating the engineering decision logic under stress in Phase 8, this phase focuses on software architecture, modularization, and operational usability.

The system is restructured to reflect real-world engineering software design principles, with clear separation of responsibilities and a centralized execution pipeline.

### What was done
- Refactored the project into logical modules (`core`, `pipeline`)
- Isolated responsibilities for data loading, decision logic, and validation
- Implemented a single orchestration script to run the full system end-to-end
- Ensured reproducibility and clean execution without manual intervention

### Final architecture

src/
├─ core/
│  ├─ data_loader.py        # Data loading and preprocessing
│  ├─ decision_engine.py    # Engineering verdict logic
│  ├─ validation_engine.py  # Degraded and extreme scenario generation
│
├─ pipeline/
│  ├─ run_full_pipeline.py  # Full system orchestration

### Pipeline behavior
The automated pipeline performs the following steps:
1. Load the structural dataset
2. Generate engineering verdicts for:
   - Base scenario
   - Degraded scenario
   - Extreme scenario
3. Aggregate all results into a single engineering report

### Output
The pipeline generates a final, unified report:

Each column represents one scenario (Base / Degraded / Extreme), and each row contains the corresponding engineering verdict:
- Safe
- Borderline
- Risky

### Run the full system

```bash
python -m src.pipeline.run_full_pipeline
