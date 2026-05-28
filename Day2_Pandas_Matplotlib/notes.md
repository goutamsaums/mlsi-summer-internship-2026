
# Day 2 Notes - Pandas and Matplotlib Fundamentals

## Introduction

Today I worked on Pandas and Matplotlib fundamentals as part of the MLSI Lab Summer Internship preparation tasks.

The main focus today was learning how to work with tabular datasets using Pandas and how to visualize data using Matplotlib. I practiced data loading, cleaning, filtering, grouping, handling missing values, and different types of plots.

I also explored plotting techniques such as line plots, scatter plots, histograms, time series plots, and subplot customization using Matplotlib tutorials.

---

# Topics Learned

## 1. Introduction to Pandas

* Pandas is used for data analysis and data manipulation.
* DataFrames help organize structured tabular data.
* Pandas simplifies cleaning, filtering, grouping, and transforming datasets.
* CSV files can be loaded directly into DataFrames.

---

# Loading CSV Data

Practiced loading datasets using Pandas.

```python
import pandas as pd

data = pd.read_csv("sample_data.csv")

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

* Shape of datasets
* Column names
* Statistical summaries
* Basic dataset understanding

---

# Handling Missing Values

Practiced handling null and missing values.

```python
print(data.isnull().sum())

cleaned = data.dropna()

filled = data.fillna(0)

interpolated = data.interpolate()
```

### Learned

* Detecting missing values
* Dropping missing rows
* Filling missing values
* Interpolation techniques

---

# Filtering and Query Operations

```python
filtered = data[data["Age"] > 25]

query_data = data.query("Salary > 50000")
```

### Learned

* Conditional filtering
* Query-based selection
* Boolean indexing
* Extracting subsets of data

---

# Group By Operations

```python
grouped = data.groupby("Department")["Salary"].mean()

print(grouped)
```

### Learned

* Grouping datasets
* Aggregation operations
* Mean, sum, count calculations
* Category-wise analysis

---

# Merge and Join Operations

```python
df1 = pd.DataFrame({
    "ID": [1, 2, 3],
    "Name": ["A", "B", "C"]
})

df2 = pd.DataFrame({
    "ID": [1, 2, 3],
    "Score": [90, 85, 88]
})

merged = pd.merge(df1, df2, on="ID")
```

### Learned

* Combining datasets
* Merge operations
* Joining tables using keys
* Relational dataset concepts

---

# Data Cleaning Pipeline

Worked on cleaning datasets before analysis.

---

# Removing Duplicates

```python
data = data.drop_duplicates()
```

### Learned

* Duplicate removal
* Data consistency improvement

---

# Outlier Detection

```python
q1 = data["Salary"].quantile(0.25)

q3 = data["Salary"].quantile(0.75)

iqr = q3 - q1
```

### Learned

* Interquartile range method
* Detecting abnormal values
* Importance of clean datasets

---

# Data Type Conversion

```python
data["Age"] = data["Age"].astype(int)
```

### Learned

* Converting data types
* Integer and float conversion
* Preparing data for analysis

---

# Creating Derived Columns

```python
data["Bonus"] = data["Salary"] * 0.10
```

### Learned

* Feature engineering basics
* Creating new columns
* Derived calculations

---

# Handling Datetime Data

```python
data["JoinDate"] = pd.to_datetime(data["JoinDate"])
```

### Learned

* Datetime conversion
* Time series preparation
* Date formatting

---

# Exporting Cleaned Data

```python
data.to_csv("cleaned_data.csv", index=False)
```

### Learned

* Exporting processed datasets
* Saving cleaned files

---

# Introduction to Matplotlib

* Matplotlib is used for data visualization.
* Helps in plotting graphs and charts.
* Useful for analysis and presentation of datasets.

---

# Line Plots

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]

y = [10, 20, 15, 30]

plt.plot(x, y)

plt.show()
```

### Learned

* Plotting line graphs
* Axes handling
* Visualizing trends

---

# Multiple Line Series

```python
plt.plot(x, y, label="Series 1")

plt.plot(x, [5, 15, 25, 35], label="Series 2")

plt.legend()
```

### Learned

* Multiple line plotting
* Legends
* Comparing datasets

---

# Scatter Plots

Practiced scatter plots based on Corey Schafer Matplotlib tutorials.

```python
x = [1, 2, 3, 4]

y = [5, 10, 15, 20]

plt.scatter(x, y)
```

### Learned

* Scatter plots show relationships between variables
* Useful for correlation analysis
* Marker customization

---

# Scatter Plot Customization

```python
plt.scatter(
    x,
    y,
    s=100,
    c="green",
    edgecolor="black",
    alpha=0.75
)
```

### Learned

* Marker size
* Colors
* Transparency
* Edge customization

---

# Histograms

Practiced histogram plotting using age distribution examples.

```python
ages = [18, 21, 25, 30, 35, 40, 45]

plt.hist(ages, bins=[10,20,30,40,50])
```

### Learned

* Distribution visualization
* Binning data
* Frequency analysis

---

# Histogram Customization

```python
plt.hist(
    ages,
    bins=[10,20,30,40,50],
    edgecolor="black"
)
```

### Learned

* Bin customization
* Improved readability
* Distribution interpretation

---

# Time Series Plotting

Worked with date-based plotting using Matplotlib.

```python
from datetime import datetime

dates = [
    datetime(2026, 6, 1),
    datetime(2026, 6, 2)
]

values = [100, 120]

plt.plot_date(dates, values)
```

### Learned

* Date-based plotting
* Time series visualization
* Plot formatting for dates

---

# Formatting Dates

```python
plt.gcf().autofmt_xdate()
```

### Learned

* Automatic date formatting
* Improving readability

---

# Subplots and Figure Customization

```python
fig, ax = plt.subplots(2, 1)

ax[0].plot(x, y)

ax[1].scatter(x, y)
```

### Learned

* Multiple plots in one figure
* Figure organization
* Axes handling

---

# Figure Titles and Labels

```python
plt.title("Sample Plot")

plt.xlabel("X Axis")

plt.ylabel("Y Axis")
```

### Learned

* Graph labeling
* Plot readability
* Proper presentation

---

# Saving High Quality Figures

```python
plt.savefig("plot.png", dpi=300)
```

### Learned

* Saving figures
* High-resolution plots
* Exporting visualizations

---

# Real-Time Plotting Concepts

Learned basics of real-time plotting from Matplotlib animation tutorials.

```python
from matplotlib.animation import FuncAnimation
```

### Learned

* Dynamic plotting
* Animation concepts
* Updating plots continuously

---

# Files Created

* dataframe_operations.py
* data_cleaning.py
* visualization_practice.py
* sample_data.csv
* notes.md

---

# Resources Used

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

!python visualization_practice.py
```

---

# Google Colab Notebook

Colab notebook used for practice and execution:

https://colab.research.google.com/drive/1ERhmqy4e08NPMO5F7jV82Pc8qaXzrEVY?usp=sharing

---

# Understanding After Day 2

Pandas makes handling structured datasets much easier compared to manual processing in Python.

Matplotlib helps visualize data clearly using different types of plots like line plots, scatter plots, histograms, and time series graphs.

I understood how important data cleaning is before analysis and how visualization helps identify patterns, trends, and outliers in datasets.

I still need more practice with advanced plotting customization and larger real-world datasets, but the basics are becoming much clearer now.
