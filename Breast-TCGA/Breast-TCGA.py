# ============================================================
# TCGA BRCA MULTI-OMICS COMPLETE EDA PIPELINE
# AUTO-LOAD FROM GITHUB + DISPLAY-ONLY
# MLSI Internship - Data Analysis Project
# 
# HOW TO USE IN GOOGLE COLAB:
# ============================================================
# Step 1: Clone the repository
# !git clone https://github.com/goutamsaums/mlsi-summer-internship-2026.git
# 
# Step 2: Navigate to the directory
# %cd /content/mlsi-summer-internship-2026/Breast-TCGA
# 
# Step 3: Run this file
# !python tcga_eda_pipeline.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore, f_oneway, chi2_contingency, pearsonr, anderson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import requests
from io import StringIO
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 1000)

# ============================================================
# CONFIGURATION - NO NEED TO CHANGE
# ============================================================

# GitHub repository base URL (raw files)
# This automatically points to your repository
GITHUB_REPO = "goutamsaums/mlsi-summer-internship-2026"
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/Breast-TCGA/"

# File mapping - matches your repository exactly
FILE_NAMES = {
    "mrna": "context1_GE.csv",
    "meth": "context2_Meth.csv",
    "mirna": "context3_miRNA.csv",
    "protein": "context4_Protein.csv",
    "clinical": "Table1Nature.csv"
}

print("="*80)
print("TCGA BRCA MULTI-OMICS EDA PIPELINE")
print("="*80)

print(f"\n📁 Repository: {GITHUB_REPO}")
print(f"📁 Data Path: Breast-TCGA/")
print(f"🔗 Base URL: {BASE_URL}")

# ============================================================
# SECTION 0: AUTO-LOAD DATA FROM GITHUB
# ============================================================

print("\n" + "="*80)
print("LOADING DATA FROM GITHUB REPOSITORY")
print("="*80)

def load_data_from_github(filename):
    """Load a CSV file directly from GitHub repository."""
    url = BASE_URL + filename
    try:
        print(f"📥 Loading: {filename} from GitHub...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        file_size = len(response.text)
        print(f"✅ Loaded: {filename} ({file_size:,} bytes)")
        return pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

# Load all files
print("\n📊 Starting data download...")
data_files = {}

for key, filename in FILE_NAMES.items():
    data_files[key] = load_data_from_github(filename)

# Check if all files loaded successfully
if any(df is None for df in data_files.values()):
    print("\n⚠️ Some files failed to load. Please check:")
    print("  1. Filenames in the repository match the ones in FILE_NAMES")
    print("  2. Repository is public or you have access")
    print("  3. Internet connection is working")
    print("\nFiles found in your repository should be:")
    for filename in FILE_NAMES.values():
        print(f"   - {filename}")
    sys.exit(1)

# Assign to variables
mrna_raw = data_files["mrna"]
meth_raw = data_files["meth"]
mirna_raw = data_files["mirna"]
protein_raw = data_files["protein"]
clinical_raw = data_files["clinical"]

print("\n✅ All files loaded successfully from GitHub!")

# ============================================================
# SECTION 1: DATA QUALITY ASSESSMENT
# ============================================================

print("\n" + "="*80)
print("SECTION 1: DATA QUALITY ASSESSMENT")
print("="*80)

# 1.1 TCGA ID HARMONIZATION
# ============================================================

def tcga_patient_id(x):
    """Convert any TCGA ID format to standard: TCGA-XX-XXXX"""
    x = str(x).strip()
    x = x.replace(".", "-")
    x = x.upper()
    parts = x.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return np.nan

def process_omics(df, name):
    print(f"\n📊 Processing {name}")
    print("-" * 50)
    
    feature_names = df.iloc[:, 0].astype(str)
    sample_ids = [tcga_patient_id(x) for x in df.columns[1:]]
    
    valid_mask = pd.Series(sample_ids).notna()
    valid_indices = valid_mask[valid_mask].index.tolist()
    
    X = df.iloc[:, 1:].T
    X = X.iloc[valid_indices]
    X.index = [sample_ids[i] for i in valid_indices]
    X.columns = feature_names
    
    X = X.groupby(X.index).mean()
    X = X.apply(pd.to_numeric, errors="coerce")
    
    print(f"✅ Patients: {X.shape[0]}, Features: {X.shape[1]}")
    return X

print("\n" + "="*60)
print("PROCESSING OMICS DATASETS")
print("="*60)

mrna = process_omics(mrna_raw, "mRNA")
meth = process_omics(meth_raw, "Methylation")
mirna = process_omics(mirna_raw, "miRNA")
protein = process_omics(protein_raw, "Protein")

# Process Clinical
print("\n📋 Processing Clinical Data")
print("-" * 50)

candidate_cols = ["Complete TCGA ID", "Patient ID", "Patient_ID", "bcr_patient_barcode", "patient"]
id_col = next((c for c in candidate_cols if c in clinical_raw.columns), None)
if id_col is None:
    print(f"⚠️ Available columns: {clinical_raw.columns.tolist()}")
    raise ValueError("No patient ID column found")

clinical_raw["patient_id"] = clinical_raw[id_col].astype(str).apply(tcga_patient_id)
clinical_raw = clinical_raw[~clinical_raw["patient_id"].isna()]
clinical_raw = clinical_raw.drop_duplicates(subset="patient_id").set_index("patient_id")
print(f"✅ Clinical Patients: {clinical_raw.shape[0]}")

# 1.2 PATIENT OVERLAP
# ============================================================

print("\n" + "="*60)
print("PATIENT OVERLAP CHECK")
print("="*60)

common_patients = sorted(
    set(mrna.index) & set(meth.index) & set(mirna.index) &
    set(protein.index) & set(clinical_raw.index)
)

overlap_data = {
    "Modality": ["mRNA", "Methylation", "miRNA", "Protein", "Clinical"],
    "Patients": [len(mrna), len(meth), len(mirna), len(protein), len(clinical_raw)]
}
overlap_df = pd.DataFrame(overlap_data)
print("\n📊 Patient Counts per Modality:")
print(overlap_df.to_string(index=False))

print(f"\n✅ COMMON PATIENTS: {len(common_patients)}")

if len(common_patients) < 100:
    print("⚠️ WARNING: Less than 100 common patients!")

mrna = mrna.loc[common_patients]
meth = meth.loc[common_patients]
mirna = mirna.loc[common_patients]
protein = protein.loc[common_patients]
clinical_matched = clinical_raw.loc[common_patients]

datasets = {
    "mRNA": mrna,
    "Methylation": meth,
    "miRNA": mirna,
    "Protein": protein
}

# ============================================================
# 1.3 MISSING VALUES ANALYSIS
# ============================================================

print("\n" + "="*60)
print("1.3 MISSING VALUES ANALYSIS")
print("="*60)

missing_summary = []

for name, df in datasets.items():
    total_missing = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percent = (total_missing / total_cells) * 100
    
    missing_per_feature = df.isna().sum()
    features_with_missing = missing_per_feature[missing_per_feature > 0]
    
    missing_per_patient = df.isna().sum(axis=1)
    patients_with_missing = missing_per_patient[missing_per_patient > 0]
    
    missing_summary.append([
        name,
        total_missing,
        f"{missing_percent:.2f}%",
        len(features_with_missing),
        len(patients_with_missing),
        missing_per_feature.max() if len(features_with_missing) > 0 else 0,
        missing_per_patient.max() if len(patients_with_missing) > 0 else 0
    ])
    
    # Missingness heatmap
    plt.figure(figsize=(12, 4))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title(f"{name} - Missing Values Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Patients")
    plt.tight_layout()
    plt.show()

missing_df = pd.DataFrame(
    missing_summary,
    columns=[
        "Modality", "TotalMissing", "MissingPercent",
        "FeaturesWithMissing", "PatientsWithMissing",
        "MaxMissingPerFeature", "MaxMissingPerPatient"
    ]
)
print("\n📊 MISSING VALUES SUMMARY:")
print(missing_df.to_string(index=False))

# ============================================================
# 1.4 OUTLIER ANALYSIS
# ============================================================

print("\n" + "="*60)
print("1.4 OUTLIER ANALYSIS")
print("="*60)

outlier_summary = []

for name, df in datasets.items():
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    outliers_iqr = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    total_outliers_iqr = outliers_iqr.sum().sum()
    outlier_percent_iqr = (total_outliers_iqr / (df.shape[0] * df.shape[1])) * 100
    
    z_scores = np.abs(zscore(df, nan_policy="omit"))
    outliers_z = z_scores > 3
    total_outliers_z = outliers_z.sum()
    outlier_percent_z = (total_outliers_z / (df.shape[0] * df.shape[1])) * 100
    
    outlier_summary.append([
        name,
        total_outliers_iqr,
        f"{outlier_percent_iqr:.2f}%",
        total_outliers_z,
        f"{outlier_percent_z:.2f}%"
    ])
    
    # Boxplot
    plt.figure(figsize=(14, 5))
    sample_features = np.random.choice(df.columns, size=min(20, len(df.columns)), replace=False)
    sns.boxplot(data=df[sample_features])
    plt.xticks(rotation=90)
    plt.title(f"{name} - Boxplot (Sample of 20 Features)")
    plt.tight_layout()
    plt.show()

outlier_df = pd.DataFrame(
    outlier_summary,
    columns=[
        "Modality", "Outliers_IQR", "OutlierPercent_IQR",
        "Outliers_ZScore", "OutlierPercent_ZScore"
    ]
)
print("\n📊 OUTLIER SUMMARY:")
print(outlier_df.to_string(index=False))

# ============================================================
# 1.5 DATA DISTRIBUTION
# ============================================================

print("\n" + "="*60)
print("1.5 DATA DISTRIBUTION")
print("="*60)

for name, df in datasets.items():
    values = df.values.flatten()
    values = values[~np.isnan(values)]
    
    if len(values) > 10000:
        values = np.random.choice(values, size=10000, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Histogram
    axes[0, 0].hist(values, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f"{name} - Histogram")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    
    # 2. KDE Plot
    sns.kdeplot(values, fill=True, ax=axes[0, 1])
    axes[0, 1].set_title(f"{name} - KDE Plot")
    axes[0, 1].set_xlabel("Value")
    
    # 3. Density Plot
    axes[0, 2].hist(values, bins=50, density=True, alpha=0.5, edgecolor='black')
    sns.kdeplot(values, ax=axes[0, 2])
    axes[0, 2].set_title(f"{name} - Density Plot")
    axes[0, 2].set_xlabel("Value")
    axes[0, 2].set_ylabel("Density")
    
    # 4. Q-Q Plot
    stats.probplot(values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f"{name} - Q-Q Plot")
    
    # 5. Anderson-Darling Test
    try:
        result = anderson(values, dist='norm')
        axes[1, 1].text(0.5, 0.5, 
                       f"Anderson-Darling Test\nStatistic: {result.statistic:.4f}\n"
                       f"5% Critical: {result.critical_values[2]:.4f}\n"
                       f"Normal at 5%: {'✅ YES' if result.statistic < result.critical_values[2] else '❌ NO'}",
                       ha='center', va='center', fontsize=12)
    except:
        axes[1, 1].text(0.5, 0.5, "Anderson-Darling\nTest unavailable", ha='center', va='center', fontsize=12)
    axes[1, 1].set_title(f"{name} - Normality Test")
    axes[1, 1].axis('off')
    
    # 6. Summary Statistics
    stats_text = f"""
    Mean: {np.mean(values):.4f}
    Median: {np.median(values):.4f}
    Std: {np.std(values):.4f}
    Skewness: {stats.skew(values):.4f}
    Kurtosis: {stats.kurtosis(values):.4f}
    """
    axes[1, 2].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12)
    axes[1, 2].set_title(f"{name} - Summary Statistics")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"✅ {name} distribution analysis displayed")

# ============================================================
# SECTION 2: FEATURE-LEVEL ANALYSIS
# ============================================================

print("\n" + "="*80)
print("SECTION 2: FEATURE-LEVEL ANALYSIS")
print("="*80)

# ============================================================
# 2.1 RANGE ANALYSIS
# ============================================================

print("\n" + "="*60)
print("2.1 RANGE ANALYSIS")
print("="*60)

range_summary = []

for name, df in datasets.items():
    mean_vals = df.mean()
    std_vals = df.std()
    cv_vals = std_vals / (mean_vals.abs() + 1e-8)
    
    feature_stats = pd.DataFrame({
        "Feature": df.columns,
        "Min": df.min(),
        "Max": df.max(),
        "Range": df.max() - df.min(),
        "Mean": mean_vals,
        "Median": df.median(),
        "Std": std_vals,
        "CV": cv_vals
    })
    feature_stats = feature_stats.replace([np.inf, -np.inf], np.nan)
    
    range_summary.append([
        name,
        df.shape[1],
        f"{feature_stats['Min'].min():.4f}",
        f"{feature_stats['Max'].max():.4f}",
        f"{feature_stats['Range'].mean():.4f}",
        f"{feature_stats['Mean'].mean():.4f}",
        f"{feature_stats['Std'].mean():.4f}",
        f"{feature_stats['CV'].mean():.4f}"
    ])
    
    print(f"\n📊 {name} - Top 10 Features by Range:")
    top_range = feature_stats.nlargest(10, 'Range')[['Feature', 'Min', 'Max', 'Range', 'Mean', 'Std']]
    print(top_range.to_string(index=False))
    
    plt.figure(figsize=(10, 5))
    plt.hist(feature_stats["Range"].dropna(), bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(feature_stats["Range"].mean(), color='red', linestyle='--', label=f'Mean: {feature_stats["Range"].mean():.4f}')
    plt.xlabel("Range")
    plt.ylabel("Number of Features")
    plt.title(f"{name} - Feature Range Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

range_df = pd.DataFrame(
    range_summary,
    columns=[
        "Modality", "TotalFeatures", "GlobalMin", "GlobalMax",
        "MeanRange", "MeanMean", "MeanStd", "MeanCV"
    ]
)
print("\n📊 RANGE ANALYSIS SUMMARY:")
print(range_df.to_string(index=False))

# ============================================================
# 2.2 DISTRIBUTION ANALYSIS
# ============================================================

print("\n" + "="*60)
print("2.2 DISTRIBUTION ANALYSIS (Skewness & Kurtosis)")
print("="*60)

distribution_summary = []

for name, df in datasets.items():
    skewness = df.skew()
    kurtosis = df.kurtosis()
    
    symmetric = (abs(skewness) < 0.5).sum()
    right_skewed = (skewness >= 0.5).sum()
    left_skewed = (skewness <= -0.5).sum()
    
    distribution_summary.append([
        name,
        len(skewness),
        f"{skewness.mean():.4f}",
        f"{skewness.median():.4f}",
        symmetric,
        right_skewed,
        left_skewed,
        f"{kurtosis.mean():.4f}",
        f"{kurtosis.median():.4f}"
    ])
    
    skew_df = pd.DataFrame({"Feature": df.columns, "Skewness": skewness})
    print(f"\n📊 {name} - Top 5 Most Skewed Features:")
    print(skew_df.nlargest(5, 'Skewness').to_string(index=False))
    print(f"\n📊 {name} - Top 5 Most Negatively Skewed Features:")
    print(skew_df.nsmallest(5, 'Skewness').to_string(index=False))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(skewness, bins=40, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', label='Symmetric')
    axes[0].axvline(skewness.mean(), color='blue', linestyle='--', label=f'Mean: {skewness.mean():.3f}')
    axes[0].set_xlabel("Skewness")
    axes[0].set_ylabel("Number of Features")
    axes[0].set_title(f"{name} - Skewness Distribution")
    axes[0].legend()
    
    axes[1].hist(kurtosis, bins=40, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='Normal (0)')
    axes[1].axvline(kurtosis.mean(), color='blue', linestyle='--', label=f'Mean: {kurtosis.mean():.3f}')
    axes[1].set_xlabel("Kurtosis")
    axes[1].set_ylabel("Number of Features")
    axes[1].set_title(f"{name} - Kurtosis Distribution")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

distribution_df = pd.DataFrame(
    distribution_summary,
    columns=[
        "Modality", "TotalFeatures", "MeanSkewness", "MedianSkewness",
        "Symmetric", "RightSkewed", "LeftSkewed",
        "MeanKurtosis", "MedianKurtosis"
    ]
)
print("\n📊 DISTRIBUTION ANALYSIS SUMMARY:")
print(distribution_df.to_string(index=False))

# ============================================================
# 2.3 CORRELATION ANALYSIS
# ============================================================

print("\n" + "="*60)
print("2.3 CORRELATION ANALYSIS (Within Modality)")
print("="*60)

correlation_summary = []

for name, df in datasets.items():
    print(f"\n📊 Analyzing: {name}")
    
    variances = df.var().replace([np.inf, -np.inf], np.nan).dropna()
    top_features = variances.sort_values(ascending=False).head(50).index
    
    corr = df[top_features].corr()
    
    # Use original correlations, not absolute
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_values = upper.stack().dropna()
    abs_corr_values = corr_values.abs()
    
    correlation_summary.append([
        name,
        f"{corr_values.mean():.4f}",
        f"{abs_corr_values.mean():.4f}",
        f"{corr_values.max():.4f}",
        (corr_values > 0.8).sum(),
        (corr_values < -0.8).sum(),
        (abs_corr_values > 0.8).sum()
    ])
    
    # Display highly correlated pairs
    high_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_stacked = high_corr.stack()
    
    if len(high_corr_stacked) > 0:
        high_corr_df = pd.DataFrame({
            'Feature1': high_corr_stacked.index.get_level_values(0),
            'Feature2': high_corr_stacked.index.get_level_values(1),
            'Correlation': high_corr_stacked.values
        })
        high_corr_df = high_corr_df[abs(high_corr_df["Correlation"]) > 0.8].sort_values("Correlation", ascending=False)
        
        if len(high_corr_df) > 0:
            print(f"\n🔗 Top 10 Highly Correlated Pairs (|r| > 0.8):")
            print(high_corr_df.head(10).to_string(index=False))
        else:
            print("No highly correlated pairs found (|r| > 0.8)")
    else:
        print("No correlation pairs available")
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title(f"{name} - Correlation Heatmap (Top 50 Features)")
    plt.tight_layout()
    plt.show()
    
    # Correlation distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(corr_values.mean(), color='red', linestyle='--', label=f'Mean: {corr_values.mean():.4f}')
    axes[0].set_xlabel("Correlation")
    axes[0].set_ylabel("Feature Pairs")
    axes[0].set_title(f"{name} - Original Correlation Distribution")
    axes[0].legend()
    
    axes[1].hist(abs_corr_values, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(abs_corr_values.mean(), color='red', linestyle='--', label=f'Mean: {abs_corr_values.mean():.4f}')
    axes[1].set_xlabel("Absolute Correlation")
    axes[1].set_ylabel("Feature Pairs")
    axes[1].set_title(f"{name} - Absolute Correlation Distribution")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

correlation_df = pd.DataFrame(
    correlation_summary,
    columns=[
        "Modality", "MeanCorrelation", "MeanAbsCorrelation",
        "MaxCorrelation", "StrongPositive", "StrongNegative", "StrongAbs"
    ]
)
print("\n📊 CORRELATION SUMMARY:")
print(correlation_df.to_string(index=False))

# ============================================================
# SECTION 3: BIVARIATE ANALYSIS
# ============================================================

print("\n" + "="*80)
print("SECTION 3: BIVARIATE ANALYSIS")
print("="*80)

# Labels
if "PAM50 mRNA" in clinical_matched.columns:
    y = clinical_matched["PAM50 mRNA"]
else:
    print("⚠️ PAM50 mRNA not found. Using first clinical column as label.")
    y = clinical_matched.iloc[:, 0]

# ============================================================
# 3.1 PAM50 DISTRIBUTION
# ============================================================

print("\n" + "="*60)
print("3.1 PAM50 SUBTYPE DISTRIBUTION")
print("="*60)

if "PAM50 mRNA" in clinical_matched.columns:
    subtype_counts = clinical_matched["PAM50 mRNA"].value_counts()
    subtype_pct = (subtype_counts / len(clinical_matched)) * 100
    
    pam50_df = pd.DataFrame({
        "Subtype": subtype_counts.index,
        "Count": subtype_counts.values,
        "Percentage": subtype_pct.values
    })
    print("\n📊 PAM50 Subtype Distribution:")
    print(pam50_df.to_string(index=False))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].pie(subtype_counts.values, labels=subtype_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("PAM50 Subtype Distribution (Pie)")
    
    axes[1].bar(subtype_counts.index, subtype_counts.values, color='skyblue', edgecolor='black')
    axes[1].set_title("PAM50 Subtype Distribution (Bar)")
    axes[1].set_xlabel("Subtype")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# 3.2 CLINICAL-CLINICAL RELATIONSHIPS
# ============================================================

print("\n" + "="*60)
print("3.2 CLINICAL-CLINICAL RELATIONSHIPS")
print("="*60)

clinical_vars = ["ER Status", "PR Status", "HER2 Final Status", "AJCC Stage", "Tumor", "Node"]
available_clinical = [c for c in clinical_vars if c in clinical_matched.columns]

if len(available_clinical) >= 2:
    clinical_association_results = []
    
    for i in range(len(available_clinical)):
        for j in range(i+1, len(available_clinical)):
            var1, var2 = available_clinical[i], available_clinical[j]
            
            ct = pd.crosstab(clinical_matched[var1], clinical_matched[var2])
            
            try:
                chi2, p, dof, expected = chi2_contingency(ct)
                clinical_association_results.append([var1, var2, f"{chi2:.4f}", f"{p:.4e}", p < 0.05])
            except:
                clinical_association_results.append([var1, var2, "N/A", "N/A", False])
            
            print(f"\n📊 {var1} vs {var2}:")
            print(ct)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd")
            plt.title(f"{var1} vs {var2}")
            plt.tight_layout()
            plt.show()
    
    clinical_association_df = pd.DataFrame(
        clinical_association_results,
        columns=["Variable1", "Variable2", "ChiSquare", "PValue", "Significant"]
    )
    print("\n📊 CLINICAL ASSOCIATION SUMMARY:")
    print(clinical_association_df.to_string(index=False))

# ============================================================
# 3.3 CLINICAL-OMICS RELATIONSHIPS
# ============================================================

print("\n" + "="*60)
print("3.3 CLINICAL-OMICS RELATIONSHIPS")
print("="*60)

# Age vs Omics with correlation
age_col = None
possible_age = ["Age at Initial Pathologic Diagnosis", "Age", "age"]
for col in possible_age:
    if col in clinical_matched.columns:
        age_col = col
        break

if age_col is not None:
    age = pd.to_numeric(clinical_matched[age_col], errors="coerce")
    age_correlations = []
    
    for name, df in datasets.items():
        top_features = df.var().sort_values(ascending=False).head(5).index
        
        for feature in top_features:
            mask = ~(age.isna() | df[feature].isna())
            if mask.sum() > 3:
                r, p = pearsonr(age[mask], df[feature][mask])
                age_correlations.append([name, feature, f"{r:.4f}", f"{p:.4e}", p < 0.05])
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=age, y=df[feature], alpha=0.6)
            plt.xlabel("Age")
            plt.ylabel(feature)
            plt.title(f"{name} - Age vs {feature}")
            plt.tight_layout()
            plt.show()
    
    age_corr_df = pd.DataFrame(
        age_correlations,
        columns=["Modality", "Feature", "Correlation", "PValue", "Significant"]
    )
    print("\n📊 AGE-OMICS CORRELATIONS:")
    print(age_corr_df.to_string(index=False))

# PAM50 vs Omics
if "PAM50 mRNA" in clinical_matched.columns:
    pam50_anova_results = []
    
    for name, df in datasets.items():
        top_features = df.var().sort_values(ascending=False).head(5).index
        
        print(f"\n📊 {name} - PAM50 vs Top Features:")
        
        for feature in top_features:
            groups = []
            for subtype in y.unique():
                group = df.loc[y == subtype, feature].dropna()
                if len(group) > 0:
                    groups.append(group)
            
            if len(groups) >= 2:
                try:
                    f_stat, p_val = f_oneway(*groups)
                    pam50_anova_results.append([name, feature, f"{f_stat:.4f}", f"{p_val:.4e}", p_val < 0.05])
                except:
                    pass
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                sns.boxplot(x=y, y=df[feature], ax=axes[i])
                axes[i].set_title(feature[:30])
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f"{name} - PAM50 vs Top 5 Features", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    pam50_df = pd.DataFrame(
        pam50_anova_results,
        columns=["Modality", "Feature", "FStatistic", "PValue", "Significant"]
    )
    print("\n📊 PAM50 ANOVA RESULTS:")
    print(pam50_df.to_string(index=False))

# ============================================================
# 3.4 OMICS-OMICS RELATIONSHIPS
# ============================================================

print("\n" + "="*60)
print("3.4 OMICS-OMICS RELATIONSHIPS")
print("="*60)

top_mrna = mrna.var().sort_values(ascending=False).head(3).index
top_protein = protein.var().sort_values(ascending=False).head(3).index

omics_correlations = []

print("\n📊 mRNA vs Protein Scatter Plots:")
for gene in top_mrna:
    for protein_feat in top_protein:
        mask = ~(mrna[gene].isna() | protein[protein_feat].isna())
        if mask.sum() > 3:
            r, p = pearsonr(mrna[gene][mask], protein[protein_feat][mask])
            omics_correlations.append(["mRNA_Protein", gene, protein_feat, f"{r:.4f}", f"{p:.4e}"])
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=mrna[gene], y=protein[protein_feat], alpha=0.6)
        plt.xlabel(f"mRNA: {gene}")
        plt.ylabel(f"Protein: {protein_feat}")
        plt.title(f"mRNA vs Protein (r={r:.3f})" if 'r' in locals() else "mRNA vs Protein")
        plt.tight_layout()
        plt.show()

top_meth = meth.var().sort_values(ascending=False).head(3).index

print("\n📊 mRNA vs Methylation Scatter Plots:")
for gene in top_mrna:
    for meth_feat in top_meth:
        mask = ~(mrna[gene].isna() | meth[meth_feat].isna())
        if mask.sum() > 3:
            r, p = pearsonr(mrna[gene][mask], meth[meth_feat][mask])
            omics_correlations.append(["mRNA_Methylation", gene, meth_feat, f"{r:.4f}", f"{p:.4e}"])
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=mrna[gene], y=meth[meth_feat], alpha=0.6)
        plt.xlabel(f"mRNA: {gene}")
        plt.ylabel(f"Methylation: {meth_feat}")
        plt.title(f"mRNA vs Methylation (r={r:.3f})" if 'r' in locals() else "mRNA vs Methylation")
        plt.tight_layout()
        plt.show()

omics_corr_df = pd.DataFrame(
    omics_correlations,
    columns=["Pair", "Feature1", "Feature2", "Correlation", "PValue"]
)
print("\n📊 OMICS CROSS-CORRELATIONS:")
print(omics_corr_df.to_string(index=False))

# ============================================================
# SECTION 4: MULTIVARIATE ANALYSIS
# ============================================================

print("\n" + "="*80)
print("SECTION 4: MULTIVARIATE ANALYSIS")
print("="*80)

# ============================================================
# 4.1 PCA ANALYSIS
# ============================================================

print("\n" + "="*60)
print("4.1 PCA ANALYSIS")
print("="*60)

top_mrna = mrna.var().sort_values(ascending=False).head(500).index
top_meth = meth.var().sort_values(ascending=False).head(500).index

integrated = pd.concat([
    mrna[top_mrna],
    meth[top_meth],
    mirna,
    protein
], axis=1)

integrated = integrated.fillna(integrated.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(integrated)

pca = PCA()
pca.fit(X_scaled)

explained = pca.explained_variance_ratio_
cum_var = np.cumsum(explained)

print(f"\n📊 PCA Variance Explained:")
print(f"PC1: {explained[0]*100:.1f}%")
print(f"PC1-2: {cum_var[1]*100:.1f}%")
print(f"PC1-5: {cum_var[4]*100:.1f}%")
print(f"PC1-10: {cum_var[9]*100:.1f}%")
print(f"PC1-20: {cum_var[19]*100:.1f}%")

plt.figure(figsize=(10, 6))
plt.bar(range(1, 21), explained[:20], alpha=0.7)
plt.plot(range(1, 21), cum_var[:20], 'ro-', linewidth=2)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("PCA - Variance Explained (Top 20 Components)")
plt.legend(["Cumulative", "Individual"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.7)
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.title("PCA 2D Projection")
plt.tight_layout()
plt.show()

if "PAM50 mRNA" in clinical_matched.columns:
    plt.figure(figsize=(10, 8))
    for subtype in y.unique():
        mask = y == subtype
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=subtype, alpha=0.7)
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    plt.title("PCA 2D - Colored by PAM50 Subtype")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 4.2 CLUSTERING ANALYSIS
# ============================================================

print("\n" + "="*60)
print("4.2 CLUSTERING ANALYSIS")
print("="*60)

k_range = range(2, 11)
inertia = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_pca)
    inertia.append(kmeans.inertia_)
    try:
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append(score)
    except:
        silhouette_scores.append(0)

optimal_k = k_range[np.argmax(silhouette_scores) if len(silhouette_scores) > 0 else 0]

print(f"\n📊 Optimal k: {optimal_k}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertia, 'bo-')
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'ro-')
axes[1].axvline(optimal_k, color='green', linestyle='--', label=f'Optimal k={optimal_k}')
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score Method")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab10", alpha=0.7)
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.title(f"KMeans Clustering (k={optimal_k})")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

sample_idx = np.random.choice(X_scaled.shape[0], size=min(100, X_scaled.shape[0]), replace=False)
X_sample = X_scaled[sample_idx]

linkage_matrix = linkage(X_sample, method="ward")

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, no_labels=True, color_threshold=0.7 * max(linkage_matrix[:, 2]))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Patients")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# ============================================================
# 4.3 CROSS-MODALITY RELATIONSHIPS
# ============================================================

print("\n" + "="*60)
print("4.3 CROSS-MODALITY RELATIONSHIPS (PC1 based)")
print("="*60)

cross_modal_pc1 = {}

for name, df in datasets.items():
    df_clean = df.fillna(df.median())
    scaler = StandardScaler()
    X_scaled_mod = scaler.fit_transform(df_clean)
    pca_mod = PCA(n_components=1)
    pc1 = pca_mod.fit_transform(X_scaled_mod).flatten()
    cross_modal_pc1[name] = pc1
    print(f"  {name}: PC1 explains {pca_mod.explained_variance_ratio_[0]:.2%} variance")

cross_modal_df = pd.DataFrame(cross_modal_pc1)
cross_corr_pc1 = cross_modal_df.corr()

print("\n📊 CROSS-MODALITY CORRELATION (PC1 based):")
print(cross_corr_pc1)

plt.figure(figsize=(6, 5))
sns.heatmap(cross_corr_pc1, annot=True, cmap="coolwarm", center=0, square=True)
plt.title("Cross-Modality Correlation (PC1)")
plt.tight_layout()
plt.show()

modalities = ["mRNA", "Methylation", "miRNA", "Protein"]
for i in range(len(modalities)):
    for j in range(i+1, len(modalities)):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=cross_modal_df[modalities[i]], y=cross_modal_df[modalities[j]], alpha=0.6)
        plt.xlabel(f"{modalities[i]} PC1")
        plt.ylabel(f"{modalities[j]} PC1")
        corr_val = cross_corr_pc1.loc[modalities[i], modalities[j]]
        plt.title(f"{modalities[i]} vs {modalities[j]} (r={corr_val:.3f})")
        plt.tight_layout()
        plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*80)
print("✅ EDA PIPELINE COMPLETE")
print("="*80)

print("\n📊 EXECUTIVE SUMMARY:")
print("-" * 50)
print(f"Total Patients: {len(common_patients)}")
print(f"mRNA Features: {mrna.shape[1]}")
print(f"Methylation Features: {meth.shape[1]}")
print(f"miRNA Features: {mirna.shape[1]}")
print(f"Protein Features: {protein.shape[1]}")

if "PAM50 mRNA" in clinical_matched.columns:
    print(f"\nPAM50 Subtypes:")
    for subtype, count in clinical_matched["PAM50 mRNA"].value_counts().items():
        print(f"  {subtype}: {count} ({count/len(clinical_matched)*100:.1f}%)")

print(f"\nPCA Variance Explained:")
print(f"  PC1: {explained[0]*100:.1f}%")
print(f"  PC1-2: {cum_var[1]*100:.1f}%")
print(f"  PC1-10: {cum_var[9]*100:.1f}%")

print(f"\nOptimal Clusters (k): {optimal_k}")

print("\n" + "="*80)
print("✅ ALL ANALYSES COMPLETE - ALL OUTPUTS DISPLAYED ON SCREEN")
print("="*80)