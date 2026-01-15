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

## Phase 2 – Machine Learning Baseline (In Progress)

Phase 2 extends the project towards predictive modeling and machine learning.

### Current focus
- Dataset generation and feature combination
- Train / test splitting
- Baseline regression models
- Initial evaluation metrics

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
