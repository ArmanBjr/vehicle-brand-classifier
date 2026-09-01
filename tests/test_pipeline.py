from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_train_and_test_csv_exist_with_expected_columns():
    train = pd.read_csv(DATA / "train_data.csv", nrows=100)
    test = pd.read_csv(DATA / "test_data.csv", nrows=100)

    feature_cols = {
        "year",
        "price",
        "transmission",
        "mileage",
        "fuelType",
        "tax",
        "mpg",
        "engineSize",
    }
    assert feature_cols.issubset(train.columns)
    assert feature_cols.issubset(test.columns)
    assert "Manufacturer" in train.columns
    assert "Manufacturer" not in test.columns


def test_svm_pipeline_runs_on_sample():
    train = pd.read_csv(DATA / "train_data.csv", nrows=400)
    features = [
        "year",
        "price",
        "transmission",
        "mileage",
        "fuelType",
        "tax",
        "mpg",
        "engineSize",
    ]
    x = train[features]
    y = train["Manufacturer"]

    numeric = ["year", "price", "mileage", "tax", "mpg", "engineSize"]
    categorical = ["transmission", "fuelType"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ]
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", SVC(kernel="linear", C=1.0)),
    ])
    model.fit(x, y)
    preds = model.predict(x.iloc[:20])
    assert len(preds) == 20
