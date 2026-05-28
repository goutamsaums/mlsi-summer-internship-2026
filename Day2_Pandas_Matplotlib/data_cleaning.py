import pandas as pd
import numpy as np

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

data = pd.read_csv("employees.csv")

print("\n========== ORIGINAL DATA ==========")

print(data.head())

# ---------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------

print("\n========== DATASET INFO ==========")

print(data.info())

# ---------------------------------------------------
# CHECK MISSING VALUES
# ---------------------------------------------------

print("\n========== MISSING VALUES BEFORE CLEANING ==========")

print(data.isnull().sum())

# ---------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------

print("\n========== SHAPE BEFORE REMOVING DUPLICATES ==========")

print(data.shape)

data = data.drop_duplicates()

print("\n========== SHAPE AFTER REMOVING DUPLICATES ==========")

print(data.shape)

# ---------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------

# Fill missing Age with mean age
data["Age"] = data["Age"].fillna(data["Age"].mean())

# Fill missing Salary with median salary
data["Salary"] = data["Salary"].fillna(data["Salary"].median())

# Fill missing PerformanceScore with mean score
data["PerformanceScore"] = data["PerformanceScore"].fillna(
    data["PerformanceScore"].mean()
)

# Fill missing JoinDate
data["JoinDate"] = data["JoinDate"].fillna("2022-01-01")

# ---------------------------------------------------
# CHECK MISSING VALUES AFTER FILLING
# ---------------------------------------------------

print("\n========== MISSING VALUES AFTER CLEANING ==========")

print(data.isnull().sum())

# ---------------------------------------------------
# CONVERT DATA TYPES
# ---------------------------------------------------

data["JoinDate"] = pd.to_datetime(data["JoinDate"])

data["Age"] = data["Age"].astype(int)

data["Salary"] = data["Salary"].astype(float)

print("\n========== UPDATED DATA TYPES ==========")

print(data.dtypes)

# ---------------------------------------------------
# CREATE DERIVED COLUMNS
# ---------------------------------------------------

# Bonus column
data["Bonus"] = data["Salary"] * 0.10

# Total compensation
data["TotalCompensation"] = data["Salary"] + data["Bonus"]

# Experience level category
data["ExperienceLevel"] = np.where(
    data["Experience"] >= 7,
    "Senior",
    "Junior"
)

# Salary category
data["SalaryCategory"] = pd.cut(
    data["Salary"],
    bins=[0, 60000, 85000, 150000],
    labels=["Low", "Medium", "High"]
)

# ---------------------------------------------------
# OUTLIER DETECTION USING IQR
# ---------------------------------------------------

Q1 = data["Salary"].quantile(0.25)

Q3 = data["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

outliers = data[
    (data["Salary"] < lower_bound) |
    (data["Salary"] > upper_bound)
]

print("\n========== OUTLIERS BASED ON SALARY ==========")

print(outliers[[
    "EmployeeID",
    "Name",
    "Salary"
]])

# ---------------------------------------------------
# SORT DATA BY JOIN DATE
# ---------------------------------------------------

data = data.sort_values(by="JoinDate")

print("\n========== SORTED DATA ==========")

print(data.head())

# ---------------------------------------------------
# FILTER SENIOR EMPLOYEES
# ---------------------------------------------------

senior_employees = data[
    data["ExperienceLevel"] == "Senior"
]

print("\n========== SENIOR EMPLOYEES ==========")

print(senior_employees[[
    "Name",
    "Department",
    "Experience",
    "Salary"
]])

# ---------------------------------------------------
# EXPORT CLEANED DATA
# ---------------------------------------------------

data.to_csv(
    "cleaned_employees.csv",
    index=False
)

print("\nCleaned dataset exported successfully.")

# ---------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------

print("\n========== CLEANED DATA SUMMARY ==========")

print(data.describe())

# ---------------------------------------------------

print("\n========== DATA CLEANING COMPLETED ==========")
