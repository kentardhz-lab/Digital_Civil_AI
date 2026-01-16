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

Notes:
- Dataset is very small, so metrics (especially R²) are unstable.
- A leakage-prone feature idea was identified and avoided (target leakage awareness).
- Next step: try a non-linear model and/or cross-validation for small datasets.
