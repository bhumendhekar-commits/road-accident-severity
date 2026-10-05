
# ============================================================
# ROAD ACCIDENT SEVERITY PREDICTION
# Model Training File
# ============================================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv("accident_data.csv")

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ============================================================
# 2. CONVERT TARGET VALUES
# ============================================================

df["Accident_severity"] = df["Accident_severity"].replace({
    "Slight Injury": "Minor",
    "Serious injury": "Serious",
    "Fatal injury": "Severe"
})


print("\nAccident Severity Distribution:")
print(df["Accident_severity"].value_counts())


# ============================================================
# 3. SELECT FEATURES
# ============================================================

features = [
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Type_of_vehicle",
    "Area_accident_occured",
    "Road_surface_type",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Number_of_vehicles_involved",
    "Number_of_casualties",
    "Cause_of_accident"
]


# ============================================================
# 4. INPUT AND TARGET
# ============================================================

X = df[features]

y = df["Accident_severity"]


print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget:")
print("Accident_severity")


# ============================================================
# 5. CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Type_of_vehicle",
    "Area_accident_occured",
    "Road_surface_type",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Cause_of_accident"
]


# ============================================================
# 6. NUMERICAL FEATURES
# ============================================================

numeric_features = [
    "Number_of_vehicles_involved",
    "Number_of_casualties"
]


# ============================================================
# 7. NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


# ============================================================
# 8. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# ============================================================
# 9. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 10. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 11. CREATE COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        model
    )
])


# ============================================================
# 12. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n----------------------------------------")
print("DATA SPLIT")
print("----------------------------------------")

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n----------------------------------------")
print("TRAINING MODEL")
print("----------------------------------------")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 15. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n----------------------------------------")
print("MODEL ACCURACY")
print("----------------------------------------")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n----------------------------------------")
print("CLASSIFICATION REPORT")
print("----------------------------------------")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n----------------------------------------")
print("CONFUSION MATRIX")
print("----------------------------------------")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 18. CREATE MODEL FOLDER
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


# ============================================================
# 19. SAVE TRAINED MODEL
# ============================================================

model_path = "model/accident_severity_model.pkl"

joblib.dump(
    pipeline,
    model_path
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n----------------------------------------")
print("MODEL SAVED SUCCESSFULLY")
print("----------------------------------------")

print(
    f"Saved at: {model_path}"
)

print("\nProject training completed successfully!")

