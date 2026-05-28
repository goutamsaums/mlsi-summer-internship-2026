# Day 2 Notes - Pandas and Matplotlib Fundamentals

## Introduction

Today I worked on Pandas and Matplotlib fundamentals as part of the MLSI Lab Summer Internship preparation tasks.

The main focus today was learning how to work with structured datasets using Pandas and how to visualize data using Matplotlib. I practiced loading CSV files, inspecting DataFrames, handling missing values, filtering data, grouping operations, cleaning datasets, and generating different types of visualizations.

For practice, I created and worked with an employee dataset named `employees.csv` containing employee information such as department, salary, experience, performance scores, work hours, promotion details, and remote work status.

I also explored plotting techniques such as line plots, scatter plots, histograms, bar charts, pie charts, subplots, and customized visualizations using Matplotlib.

---

# Topics Learned

## 1. Introduction to Pandas

* Pandas is used for data analysis and manipulation.
* DataFrames organize structured tabular data efficiently.
* Pandas simplifies filtering, grouping, cleaning, and transforming datasets.
* CSV datasets can be loaded directly into DataFrames.

---

# Loading CSV Data

Practiced loading datasets using Pandas.

```python
import pandas as pd

data = pd.read_csv("employees.csv")

print(data.head())

print(data.info())
```

### Learned

* Reading CSV files
* Inspecting datasets
* Viewing column information
* Understanding rows and columns

---

# DataFrame Inspection

```python
print(data.shape)

print(data.columns)

print(data.describe())
```

### Learned

* Dataset dimensions
* Column names
* Statistical summaries
* Understanding numerical features

---

# Handling Missing Values

Practiced handling missing values in the employee dataset.

```python
print(data.isnull().sum())

data["Age"] = data["Age"].fillna(
    data["Age"].mean()
)

data["Salary"] = data["Salary"].fillna(
    data["Salary"].median()
)
```

### Learned

* Detecting null values
* Filling missing data using mean and median
* Importance of clean datasets

---

# Filtering and Query Operations

```python
filtered = data[data["Salary"] > 80000]

query_data = data.query(
    "Department == 'Data Science'"
)
```

### Learned

* Conditional filtering
* Boolean indexing
* Query-based selection
* Extracting subsets of data

---

# Group By Operations

```python
grouped = data.groupby(
    "Department"
)["Salary"].mean()

print(grouped)
```

### Learned

* Grouping datasets
* Aggregation operations
* Mean, count, and sum calculations
* Department-wise analysis

---

# Merge and Join Operations

```python
df1 = pd.DataFrame({
    "EmployeeID": [1, 2, 3],
    "Name": ["A", "B", "C"]
})

df2 = pd.DataFrame({
    "EmployeeID": [1, 2, 3],
    "Score": [90, 85, 88]
})

merged = pd.merge(
    df1,
    df2,
    on="EmployeeID"
)
```

### Learned

* Combining datasets
* Merge operations
* Joining tables using keys
* Relational dataset concepts

---

# Data Cleaning Pipeline

Worked on cleaning the employee dataset before visualization and analysis.

---

# Removing Duplicates

```python
data = data.drop_duplicates()
```

### Learned

* Removing repeated records
* Improving dataset consistency

---

# Outlier Detection

```python
q1 = data["Salary"].quantile(0.25)

q3 = data["Salary"].quantile(0.75)

iqr = q3 - q1
```

### Learned

* Detecting salary outliers
* Interquartile Range (IQR) method
* Importance of removing abnormal values

---

# Data Type Conversion

```python
data["JoinDate"] = pd.to_datetime(
    data["JoinDate"]
)
```

### Learned

* Datetime conversion
* Working with date columns
* Preparing datasets for time-based analysis

---

# Creating Derived Columns

```python
data["Bonus"] = data["Salary"] * 0.10

data["SalaryPerExperience"] = (
    data["Salary"] / data["Experience"]
)
```

### Learned

* Feature engineering basics
* Creating new analytical columns
* Derived calculations

---

# Exporting Cleaned Data

```python
data.to_csv(
    "cleaned_employees.csv",
    index=False
)
```

### Learned

* Exporting cleaned datasets
* Saving processed files

---

# Introduction to Matplotlib

* Matplotlib is used for data visualization.
* Helps in plotting graphs and charts.
* Useful for analysis and presentation of datasets.

---

# Line Plots

Plotted salary and performance score trends.

```python
plt.plot(
    data["EmployeeID"],
    data["Salary"]
)
```

### Learned

* Plotting line graphs
* Trend visualization
* Axes handling

---

# Multiple Line Series

```python
plt.plot(
    data["EmployeeID"],
    data["Salary"],
    label="Salary"
)

plt.plot(
    data["EmployeeID"],
    data["PerformanceScore"] * 10000,
    label="Performance"
)

plt.legend()
```

### Learned

* Multiple line plotting
* Legends
* Comparing multiple series

---

# Scatter Plots

Practiced scatter plots using employee experience and salary.

```python
plt.scatter(
    data["Experience"],
    data["Salary"]
)
```

### Learned

* Scatter plots show relationships between variables
* Useful for correlation analysis
* Identifying trends and outliers

---

# Scatter Plot Customization

```python
plt.scatter(
    data["Experience"],
    data["Salary"],
    c=data["PerformanceScore"],
    cmap="viridis",
    s=120,
    edgecolor="black",
    alpha=0.75
)
```

### Learned

* Marker size
* Color mapping
* Transparency
* Edge customization
* Colorbars for additional information

---

# Histograms

Plotted age distribution of employees.

```python
plt.hist(
    data["Age"],
    bins=8,
    edgecolor="black"
)
```

### Learned

* Distribution visualization
* Frequency analysis
* Understanding dataset spread

---

# Bar Charts

```python
department_counts = data["Department"].value_counts()

department_counts.plot(kind="bar")
```

### Learned

* Category visualization
* Comparing department sizes
* Aggregated visual analysis

---

# Pie Charts

```python
remote_counts = data["RemoteWork"].value_counts()

plt.pie(
    remote_counts,
    labels=remote_counts.index,
    autopct="%1.1f%%"
)
```

### Learned

* Percentage-based visualization
* Distribution comparison

---

# Subplots and Figure Customization

```python
fig, ax = plt.subplots(2, 1)

ax[0].plot(
    data["EmployeeID"],
    data["Salary"]
)

ax[1].scatter(
    data["Experience"],
    data["PerformanceScore"]
)
```

### Learned

* Multiple plots in one figure
* Figure organization
* Axes handling

---

# Figure Titles and Labels

```python
plt.title("Employee Salary Analysis")

plt.xlabel("Employee ID")

plt.ylabel("Salary")
```

### Learned

* Graph labeling
* Plot readability
* Presentation quality

---

# Saving High Quality Figures

```python
plt.savefig(
    "plot.png",
    dpi=300
)
```

### Learned

* Saving figures
* High-resolution visualization export

---

# Time Series Plotting Concepts

Studied basics of plotting date-based data using Matplotlib tutorials.

```python
plt.plot_date(dates, values)
```

### Learned

* Date-based plotting
* Time series visualization
* Date formatting concepts

---

# Real-Time Plotting Concepts

Learned basics of real-time plotting using Matplotlib animation tutorials.

```python
from matplotlib.animation import FuncAnimation
```

### Learned

* Dynamic plotting
* Live graph updates
* Animation concepts

---

# Files Created

* employees.csv
* dataframe_operations.py
* data_cleaning.py
* visualization_practice.py
* notes.md

---

# Resources Used

* Pandas tutorial video
* Google Colab for execution and testing

---

# Google Colab Practice

Used Google Colab to execute Python files directly from the GitHub repository.

Commands used:

```python
!git clone https://github.com/goutamsaums/mlsi-summer-internship-2026.git

%cd mlsi-summer-internship-2026/Day2_Pandas_Matplotlib

!python dataframe_operations.py

!python data_cleaning.py

%run visualization_practice.py
```

---

# Google Colab Notebook

Colab notebook used for practice and execution:

https://colab.research.google.com/drive/1ERhmqy4e08NPMO5F7jV82Pc8qaXzrEVY?usp=sharing

---

# Understanding After Day 2

Pandas makes working with structured datasets much easier compared to manual processing in Python.

Matplotlib helps visualize data clearly using line plots, scatter plots, histograms, bar charts, pie charts, and subplots.

I understood how important data cleaning is before analysis and how visualization helps identify trends, relationships, distributions, and outliers in datasets.

I still need more practice with advanced plotting customization and larger real-world datasets, but the basics are becoming much clearer now.
