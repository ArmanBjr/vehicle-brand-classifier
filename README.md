# Vehicle Brand Classifier

Project 1 for **Fundamentals of Computational Intelligence** at Ferdowsi University of Mashhad (FUM). Multiclass classification of car manufacturers from UK used car listing data, using **SVM** and **Decision Tree** with a leakage-safe preprocessing pipeline.

> Course report: [`report/4022262035-4021262131-AmirHoseinAbolfazli-ArmanBijari.pdf`](report/4022262035-4021262131-AmirHoseinAbolfazli-ArmanBijari.pdf)

## Results

| Model | Test Accuracy |
|---|---|
| SVM (RBF, C=500, gamma=scale) | 73.51% |
| Decision Tree (entropy, depth=30) | 81.67% |

## Structure

```
data/
├── train_data.csv
├── test_data.csv
└── official_dataset/   # per-brand reference CSVs
notebooks/
└── project.ipynb       # full pipeline (EDA → tuning → predictions)
outputs/
├── label_SVM.csv
└── label_DT.csv
report/                 # LaTeX report (PDF tracked; sources local)
```

## Quick Start

```bash
git clone https://github.com/ArmanBjr/vehicle-brand-classifier.git
cd vehicle-brand-classifier

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook notebooks/project.ipynb
```

Run all cells in `notebooks/project.ipynb`. Final predictions are written to `outputs/`.

## Stack

Python · scikit-learn · imbalanced-learn (SMOTE) · pandas · seaborn

## Authors & License

**AmirHosein Abolfazli** (4022262035) · **Arman Bijari** (4021262131) — [ArmanBjr](https://github.com/ArmanBjr)

**Professor:** Dr. Fazl Ersi

Released under the [MIT License](LICENSE).
