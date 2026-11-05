import joblib
import pandas as pd

X = pd.read_csv("data/StressLevelDataset.csv")
X = X.drop(columns=["stress_level"])

models = ["logistic_regression", "random_forest"]

for name in models:
    print("\nTesting:", name)

    model = joblib.load(f"models/{name}.joblib")

    prediction = model.predict(X.iloc[[0]])

    print("Loaded successfully: YES")
    print("Prediction:", prediction[0])
    print("Prediction probability:", model.predict_proba(X.iloc[[0]])[0])