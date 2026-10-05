import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Load data

df = pd.read_csv(
    "accident_data.csv"
)


# Convert target

def convert_severity(value):

    value = str(value).lower()

    if "slight" in value:
        return "Minor"

    if "serious" in value:
        return "Serious"

    if "fatal" in value:
        return "Severe"


df["Accident_severity"] = (
    df["Accident_severity"]
    .apply(convert_severity)
)


# Remove Time

X = df.drop(
    columns=["Accident_severity", "Time"]
)

y = df["Accident_severity"]


# Same split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# Load model

model = joblib.load(
    "model/accident_severity_model.pkl"
)


# Predict

y_pred = model.predict(
    X_test
)


# Confusion matrix

cm = confusion_matrix(
    y_test,
    y_pred,

    labels=[
        "Minor",
        "Serious",
        "Severe"
    ]
)


# Plot

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,

    annot=True,

    fmt="d",

    xticklabels=[
        "Minor",
        "Serious",
        "Severe"
    ],

    yticklabels=[
        "Minor",
        "Serious",
        "Severe"
    ]
)


plt.title(
    "Road Accident Severity - Confusion Matrix"
)

plt.xlabel(
    "Predicted Severity"
)

plt.ylabel(
    "Actual Severity"
)

plt.tight_layout()

plt.show()