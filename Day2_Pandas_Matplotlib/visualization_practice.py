import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

data = pd.read_csv("employees.csv")

# ---------------------------------------------------
# HANDLE MISSING VALUES FOR VISUALIZATION
# ---------------------------------------------------

data["Salary"] = data["Salary"].fillna(
    data["Salary"].median()
)

data["Age"] = data["Age"].fillna(
    data["Age"].mean()
)

data["PerformanceScore"] = data["PerformanceScore"].fillna(
    data["PerformanceScore"].mean()
)

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------

plt.style.use("ggplot")

# ---------------------------------------------------
# LINE PLOT
# ---------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    data["EmployeeID"],
    data["Salary"],
    label="Salary",
    linewidth=2
)

plt.plot(
    data["EmployeeID"],
    data["PerformanceScore"] * 10000,
    label="Performance Score (scaled)",
    linewidth=2
)

plt.title("Employee Salary and Performance")

plt.xlabel("Employee ID")

plt.ylabel("Values")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "line_plot.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------
# SCATTER PLOT
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

scatter = plt.scatter(
    data["Experience"],
    data["Salary"],
    c=data["PerformanceScore"],
    cmap="viridis",
    s=120,
    edgecolor="black",
    alpha=0.75
)

cbar = plt.colorbar(scatter)

cbar.set_label("Performance Score")

plt.title("Experience vs Salary")

plt.xlabel("Experience")

plt.ylabel("Salary")

plt.tight_layout()

plt.savefig(
    "scatter_plot.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------
# HISTOGRAM
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    data["Age"],
    bins=8,
    edgecolor="black"
)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "histogram.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------
# BAR CHART
# ---------------------------------------------------

department_counts = data["Department"].value_counts()

plt.figure(figsize=(8, 5))

department_counts.plot(
    kind="bar"
)

plt.title("Employees per Department")

plt.xlabel("Department")

plt.ylabel("Count")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "bar_chart.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------
# PIE CHART
# ---------------------------------------------------

remote_counts = data["RemoteWork"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    remote_counts,
    labels=remote_counts.index,
    autopct="%1.1f%%"
)

plt.title("Remote Work Distribution")

plt.tight_layout()

plt.savefig(
    "pie_chart.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------
# SUBPLOTS
# ---------------------------------------------------

fig, ax = plt.subplots(
    2,
    1,
    figsize=(10, 8)
)

# Salary subplot
ax[0].plot(
    data["EmployeeID"],
    data["Salary"]
)

ax[0].set_title("Salary Plot")

ax[0].set_xlabel("Employee ID")

ax[0].set_ylabel("Salary")

# Experience vs Performance subplot
ax[1].scatter(
    data["Experience"],
    data["PerformanceScore"]
)

ax[1].set_title("Experience vs Performance")

ax[1].set_xlabel("Experience")

ax[1].set_ylabel("Performance Score")

plt.tight_layout()

plt.savefig(
    "subplots.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------
# SALARY DISTRIBUTION BY DEPARTMENT
# ---------------------------------------------------

departments = data.groupby("Department")["Salary"].mean()

plt.figure(figsize=(8, 5))

departments.plot(
    kind="bar"
)

plt.title("Average Salary by Department")

plt.xlabel("Department")

plt.ylabel("Average Salary")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "department_salary.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------
# EXPERIENCE VS PROJECTS
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Experience"],
    data["ProjectsCompleted"],
    c=data["Salary"],
    cmap="plasma",
    s=120,
    edgecolor="black"
)

cbar = plt.colorbar()

cbar.set_label("Salary")

plt.title("Experience vs Projects Completed")

plt.xlabel("Experience")

plt.ylabel("Projects Completed")

plt.tight_layout()

plt.savefig(
    "experience_projects.png",
    dpi=300
)

plt.show()

# ---------------------------------------------------
# SAVE FINAL CLEANED VISUALIZATION DATA
# ---------------------------------------------------

data.to_csv(
    "visualization_ready_data.csv",
    index=False
)

print("\nVisualization practice completed successfully.")

