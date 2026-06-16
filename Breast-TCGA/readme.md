
# TCGA BRCA Multi-Omics EDA Pipeline

## Overview

This project implements a comprehensive **Exploratory Data Analysis (EDA) pipeline** for TCGA Breast Cancer (TCGA-BRCA) multi-omics data. The pipeline follows a systematic workflow to analyze mRNA, DNA methylation, miRNA, and protein expression data along with clinical information.

The objective was to understand the data quality, feature distributions, biological associations, and multi-omics integration patterns using Python libraries including Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and SciPy.

Topics covered:

- Data Quality Assessment
- Missing Values Analysis
- Outlier Detection (IQR Method)
- Data Distribution Analysis
- Feature-Level Statistics
- Variance Analysis
- Skewness and Kurtosis
- Correlation Analysis
- Bivariate Analysis
- Clinical-Omics Associations
- Omics-Omics Relationships
- Multivariate Analysis
- Principal Component Analysis (PCA)
- t-SNE Visualization
- Clustering (KMeans & Hierarchical)
- Cross-Modality Correlation

---

# Learning Resources

### TCGA & Cancer Genomics

- TCGA-BRCA Data Portal
- PAM50 Subtypes Paper
- TCGA BRCA Nature Paper

### Python Libraries

- Pandas Documentation
- Seaborn Documentation
- Scikit-learn Documentation
- Statsmodels Documentation

---

# Data Files

| File | Description | Features |
|------|-------------|----------|
| context1_GE.csv | mRNA expression data | 645 genes |
| context2_Meth.csv | DNA methylation data | 574 CpG sites |
| context3_miRNA.csv | miRNA expression data | 423 miRNAs |
| context4_Protein.csv | Protein expression data | 171 proteins |
| Table1Nature.csv | Clinical data | 30 clinical variables |

---

# Pipeline Structure

## 1. Data Loading & Preprocessing

The pipeline automatically loads data from the GitHub repository and performs TCGA ID harmonization.

### TCGA ID Harmonization

TCGA IDs come in various formats:

```python
# Examples of TCGA ID formats
# TCGA.A1.A0SH.01A.11R.A084.07 -> TCGA-A1-A0SH
# TCGA.A1.A0SH.01A -> TCGA-A1-A0SH
# TCGA-A1-A0SH -> TCGA-A1-A0SH
# TCGA-A1-A0SH-01A-11R-A084-07 -> TCGA-A1-A0SH

def tcga_patient_id(x):
    """Convert any TCGA ID format to standard: TCGA-XX-XXXX"""
    x = str(x).strip()
    x = x.replace(".", "-")
    x = x.upper()
    parts = x.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return np.nan
```

### Patient Overlap

Common patients across all modalities:

```text
mRNA:           348 patients
Methylation:    348 patients
miRNA:          348 patients
Protein:        348 patients
Clinical:       825 patients
COMMON PATIENTS: 348
```

## 2. Data Quality Analysis

### 2.1 Dataset Overview

| Modality | Patients | Features |
|----------|----------|----------|
| mRNA | 348 | 645 |
| Methylation | 348 | 574 |
| miRNA | 348 | 423 |
| Protein | 348 | 171 |
| Clinical | 348 | 30 |

### 2.2 Missing Values Analysis

Methods Used:
- Total missing values count
- Missing percentage per modality
- Missing values per feature
- Missingness heatmaps

### 2.3 Outlier Analysis (IQR Method)

Outliers detected using the Interquartile Range (IQR) method:

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
```

### 2.4 Data Distribution

For each modality, the following plots are generated:

1. **Histogram** - Frequency distribution
2. **KDE Plot** - Kernel Density Estimation
3. **Density Plot** - Probability density
4. **Q-Q Plot** - Normality assessment
5. **Anderson-Darling Test** - Statistical normality test

## 3. Feature-Level Analysis

### 3.1 Range Analysis

For each feature, the following statistics are computed:

- Minimum value
- Maximum value
- Range (Max - Min)
- Mean
- Median
- Standard Deviation
- Coefficient of Variation (CV)

### 3.2 Variance Analysis

Variance indicates the spread of feature values. High variance features are often biologically more informative.

### 3.3 Skewness & Kurtosis Analysis

**Skewness** measures the asymmetry of the distribution:
- Symmetric: Skewness between -0.5 and 0.5
- Right-skewed: Skewness > 0.5
- Left-skewed: Skewness < -0.5

**Kurtosis** measures the tail-heaviness of the distribution:
- Platykurtic: Kurtosis < 0 (flat distribution)
- Mesokurtic: Kurtosis between -0.5 and 0.5 (normal-like)
- Leptokurtic: Kurtosis > 0.5 (heavy-tailed)

### 3.4 Correlation Analysis

Pearson correlation matrices are computed for the top 50 variable features in each modality.

Visualizations:
1. **Correlation Heatmap** - Visual representation of correlation matrix
2. **Correlation Clustermap** - Hierarchically clustered correlation matrix
3. **Correlation Distribution** - Histogram of correlation values

## 4. Bivariate Analysis

### 4.1 PAM50 Subtype Distribution

PAM50 is a gene expression-based classification system for breast cancer subtypes.

| Subtype | Count | Percentage |
|---------|-------|------------|
| Luminal A | 154 | 44.3% |
| Luminal B | 81 | 23.3% |
| Basal-like | 66 | 19.0% |
| HER2-enriched | 42 | 12.1% |
| Normal-like | 5 | 1.4% |

### 4.2 Clinical-Clinical Relationships

Methods:
- Crosstabulation
- Chi-square Test of Independence

Clinical Variables Analyzed:
- ER Status
- PR Status
- HER2 Final Status
- AJCC Stage
- Tumor
- Node

### 4.3 Clinical-Omics Relationships

**Age vs Omics:** Scatter plots and Pearson correlation for age vs top variable features.

**PAM50 vs Omics:** ANOVA to identify features significantly associated with PAM50 subtypes.

Visualizations:
- Boxplots - Show median, quartiles, and outliers
- Violin Plots - Show full distribution shape (including density)

### 4.4 Omics-Omics Relationships

- mRNA vs Protein: Scatter plots and Pearson correlation
- mRNA vs Methylation: Scatter plots and Pearson correlation

## 5. Multivariate Analysis

### 5.1 Principal Component Analysis (PCA)

PCA is used for dimensionality reduction and variance structure analysis.

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| PC1 | ~12.5% | 12.5% |
| PC2 | ~8.3% | 20.8% |
| PC1-5 | - | ~35.2% |
| PC1-10 | - | ~48.7% |
| PC1-20 | - | ~62.1% |

Visualizations:
1. **Scree Plot** - Variance explained by each component
2. **2D Projection** - PCA plot colored by PAM50 subtype

### 5.2 t-SNE Analysis

t-SNE provides nonlinear dimensionality reduction for visualizing high-dimensional data.

Parameters:
- n_components = 2
- perplexity = 30 (or n-1 for small datasets)
- learning_rate = auto

### 5.3 Clustering Analysis

**KMeans Clustering:**
- Optimal k selection using Elbow Method and Silhouette Score
- Visualization on PCA projection

**Hierarchical Clustering:**
- Method: Ward's linkage
- Visualization: Dendrogram

### 5.4 Cross-Modality Correlation (PC1-Based)

For each modality:
1. Standardize features
2. Perform PCA
3. Extract PC1
4. Correlate PC1 scores across modalities

---

# Libraries Used

```python
pandas          # Data manipulation
numpy           # Numerical operations
matplotlib      # Visualization
seaborn         # Statistical visualization
scipy           # Statistical tests
scikit-learn    # PCA, t-SNE, clustering
statsmodels     # Multiple testing correction
requests        # HTTP requests for data loading
```

---

# How to Run

### Option 1: Google Colab (Recommended)

```python
# Clone repository
!git clone https://github.com/goutamssums/mlsi-summer-internship-2026.git

# Navigate to directory
%cd /content/mlsi-summer-internship-2026/Breast-TCGA

# Install dependencies
!pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels requests

# Run the pipeline
!python EDA-Breast_TCGA.py
```

### Option 2: Local Machine

```bash
# Clone repository
git clone https://github.com/goutamssums/mlsi-summer-internship-2026.git
cd mlsi-summer-internship-2026/Breast-TCGA

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels requests

# Run the pipeline
python EDA-Breast_TCGA.py
```

---

# Google Colab Notebook

https://colab.research.google.com/github/goutamssums/mlsi-summer-internship-2026/blob/main/Breast-TCGA/EDA-Breast_TCGA.py

---

This EDA pipeline provides a complete workflow for analyzing TCGA BRCA multi-omics data:

1. **Data Quality Assessment** - Missing values, outliers, distributions
2. **Feature-Level Analysis** - Range, variance, skewness, kurtosis, correlation
3. **Bivariate Analysis** - Clinical-Clinical, Clinical-Omics, Omics-Omics
4. **Multivariate Analysis** - PCA, t-SNE, clustering, cross-modality

