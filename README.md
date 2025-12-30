# Digital Civil AI

This repository demonstrates a Python-based data analysis pipeline for civil engineering applications, with a focus on structural elements and engineering-driven validation logic.

---
## Phase 1 – Structural Data Analysis

This phase demonstrates a basic engineering data workflow using structural element data.

### Implemented Steps
- Load data from CSV
- Detect and handle missing values
- Feature engineering (load per meter)
- Visualization of engineering metrics
- Rule-based validation of unusual loads

---

## Project Structure

```text
Digital_Civil_AI/
│
├── data/
│   └── elements.csv          # Sample structural element data
│
├── notebooks/
│   └── test.ipynb            # Phase 1 data analysis notebook
│
├── src/
│   └── testok.py             # Script-based version (Phase 1)
│
├── phase1-intero.py          # Standalone executable Phase 1 pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore
