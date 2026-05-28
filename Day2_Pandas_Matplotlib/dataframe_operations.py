import pandas as pd

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

data = pd.read_csv("employees.csv")

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== LAST 5 ROWS ==========")
print(data.tail())

# ---------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------

print("\n========== DATASET INFO ==========")
print(data.info())

print("\n========== DATASET SHAPE ==========")
print(data.shape)

print("\n========== COLUMN NAMES ==========")
print(data.columns)

print("\n========== DATA TYPES ==========")
print(data.dtypes)

# ---------------------------------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------------------------------

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(data.describe())

# ---------------------------------------------------
# CHECK MISSING VALUES
# ---------------------------------------------------

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# ---------------------------------------------------
# FILTERING OPERATIONS
# ---------------------------------------------------

print("\n========== EMPLOYEES WITH SALARY > 90000 ==========")

high_salary = data[data["Salary"] > 90000]

print(high_salary)

# ---------------------------------------------------

print("\n========== EMPLOYEES FROM DATA SCIENCE ==========")

data_science = data[data["Department"] == "Data Science"]

print(data_science)

# ---------------------------------------------------

print("\n========== EMPLOYEES WITH EXPERIENCE >= 5 ==========")

experienced = data[data["Experience"] >= 5]

print(experienced)

# ---------------------------------------------------
# QUERY OPERATIONS
# ---------------------------------------------------

print("\n========== QUERY: AGE > 30 AND SALARY > 80000 ==========")

query_result = data.query("Age > 30 and Salary > 80000")

print(query_result)

# ---------------------------------------------------
# SORTING
# ---------------------------------------------------

print("\n========== SORT BY SALARY DESCENDING ==========")

sorted_salary = data.sort_values(by="Salary", ascending=False)

print(sorted_salary.head(10))

# ---------------------------------------------------
# GROUP BY OPERATIONS
# ---------------------------------------------------

print("\n========== AVERAGE SALARY BY DEPARTMENT ==========")

avg_salary = data.groupby("Department")["Salary"].mean()

print(avg_salary)

# ---------------------------------------------------

print("\n========== AVERAGE PERFORMANCE SCORE BY DEPARTMENT ==========")

avg_performance = data.groupby("Department")["PerformanceScore"].mean()

print(avg_performance)

# ---------------------------------------------------

print("\n========== EMPLOYEE COUNT BY CITY ==========")

city_count = data.groupby("City")["EmployeeID"].count()

print(city_count)

# ---------------------------------------------------
# MULTIPLE AGGREGATIONS
# ---------------------------------------------------

print("\n========== MULTIPLE AGGREGATIONS ==========")

aggregated = data.groupby("Department").agg({
    "Salary": ["mean", "max", "min"],
    "Experience": ["mean", "max"],
    "PerformanceScore": ["mean", "max"]
})

print(aggregated)

# ---------------------------------------------------
# VALUE COUNTS
# ---------------------------------------------------

print("\n========== GENDER COUNT ==========")

print(data["Gender"].value_counts())

# ---------------------------------------------------

print("\n========== REMOTE WORK COUNT ==========")

print(data["RemoteWork"].value_counts())

# ---------------------------------------------------
# SELECT SPECIFIC COLUMNS
# ---------------------------------------------------

print("\n========== NAME AND SALARY ==========")

print(data[["Name", "Salary"]].head())

# ---------------------------------------------------
# CONDITIONAL FILTERING
# ---------------------------------------------------

print("\n========== HIGH PERFORMANCE EMPLOYEES ==========")

high_performance = data[data["PerformanceScore"] >= 9]

print(high_performance)

# ---------------------------------------------------
# UNIQUE VALUES
# ---------------------------------------------------

print("\n========== UNIQUE DEPARTMENTS ==========")

print(data["Department"].unique())

# ---------------------------------------------------

print("\n========== NUMBER OF UNIQUE CITIES ==========")

print(data["City"].nunique())

# ---------------------------------------------------
# CREATE NEW DATAFRAME
# ---------------------------------------------------

bonus_df = data[["EmployeeID", "Name", "Salary"]].copy()

bonus_df["Bonus"] = bonus_df["Salary"] * 0.10

print("\n========== BONUS DATAFRAME ==========")

print(bonus_df.head())

# ---------------------------------------------------
# MERGE OPERATION
# ---------------------------------------------------

project_data = pd.DataFrame({
    "EmployeeID": [101, 105, 110, 120],
    "ProjectName": [
        "AI Chatbot",
        "Analytics Dashboard",
        "Vision Model",
        "Recommendation System"
    ]
})

merged_data = pd.merge(
    data,
    project_data,
    on="EmployeeID",
    how="left"
)

print("\n========== MERGED DATA ==========")

print(merged_data[[
    "EmployeeID",
    "Name",
    "Department",
    "ProjectName"
]].head(10))

# ---------------------------------------------------
# SAVE FILTERED DATA
# ---------------------------------------------------

high_salary.to_csv(
    "high_salary_employees.csv",
    index=False
)

print("\nHigh salary employee data exported successfully.")

# ---------------------------------------------------

print("\n========== DATAFRAME OPERATIONS COMPLETED ==========")
