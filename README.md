# Vehicle Brand Classifier

Multiclass classification of car manufacturers from UK used car listing data.  
Built with **SVM** and **Decision Tree** as part of a Computational Intelligence course project.

## Results

| Model | Test Accuracy |
|---|---|
| SVM (RBF, C=500, gamma=scale) | 73.51% |
| Decision Tree (entropy, depth=30) | 81.67% |

## Structure

```
data/          — train/test CSVs and official reference dataset
notebooks/     — main Jupyter notebook (project.ipynb)
outputs/       — label_SVM.csv and label_DT.csv predictions
```

## Stack

Python · scikit-learn · imbalanced-learn (SMOTE) · pandas · seaborn

---

**Course:** Fundamentals of Computational Intelligence  
**Professor:** Dr. Fazl Ersi  
**Authors:** Amirhossein Abolfazli · Arman Bijari
