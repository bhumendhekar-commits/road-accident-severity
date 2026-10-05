import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("accident_data.csv")

print("Dataset loaded successfully!")


# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 4. ACCIDENT SEVERITY
# ==========================================

print("\nAccident Severity:")
print(df["Accident_severity"].value_counts())


# ==========================================
# 5. SEVERITY BAR CHART
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Accident_severity"
)

plt.title("Accident Severity Distribution")

plt.xlabel("Accident Severity")

plt.ylabel("Number of Accidents")

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()


# ==========================================
# 6. WEATHER VS SEVERITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Weather_conditions",
    hue="Accident_severity"
)

plt.title(
    "Weather Conditions vs Accident Severity"
)

plt.xlabel("Weather Conditions")

plt.ylabel("Number of Accidents")

plt.xticks(rotation=30)

plt.tight_layout()

plt.show()


# ==========================================
# 7. ROAD SURFACE VS SEVERITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Road_surface_conditions",
    hue="Accident_severity"
)

plt.title(
    "Road Surface Conditions vs Accident Severity"
)

plt.xlabel("Road Surface")

plt.ylabel("Number of Accidents")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ==========================================
# 8. VEHICLES INVOLVED
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Number_of_vehicles_involved"
)

plt.title(
    "Number of Vehicles Involved"
)

plt.xlabel("Number of Vehicles")

plt.ylabel("Number of Accidents")

plt.tight_layout()

plt.show()


# ==========================================
# 9. CASUALTIES VS SEVERITY
# ==========================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Accident_severity",
    y="Number_of_casualties"
)

plt.title(
    "Number of Casualties vs Accident Severity"
)

plt.xlabel("Accident Severity")

plt.ylabel("Number of Casualties")

plt.tight_layout()

plt.show()