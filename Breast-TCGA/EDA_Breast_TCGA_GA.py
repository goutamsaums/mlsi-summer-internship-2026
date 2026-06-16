# ============================================================
# TCGA BRCA MULTI-OMICS COMPLETE PIPELINE
# AUTO-LOAD FROM GITHUB + FULL EDA + ML + GNN READY
# MLSI Internship - Data Analysis Project
# ============================================================
# 
# HOW TO RUN:
# 1. Copy this entire file into a Google Colab cell
# 2. Or save as tcga_pipeline.py and run: python tcga_pipeline.py
# 3. All data auto-loads from GitHub
# 4. All outputs display on screen
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import zscore, f_oneway, chi2_contingency, levene, pearsonr, anderson
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, balanced_accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
import networkx as nx
import umap.umap_ as umap
import requests
from io import StringIO
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 1000)

print("="*80)
print("TCGA BRCA MULTI-OMICS COMPLETE PIPELINE")
print("AUTO-LOAD FROM GITHUB")
print("="*80)

# ============================================================
# SECTION 0: CONFIGURATION
# ============================================================

# GitHub repository base URL (raw files)
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

print(f"\n📁 Repository: {GITHUB_REPO}")
print(f"📁 Data Path: Breast-TCGA/")
print(f"🔗 Base URL: {BASE_URL}")

# ============================================================
# SECTION 0.1: AUTO-LOAD DATA FROM GITHUB
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
    raise ValueError("Data loading failed. Please fix the issues above.")

# Assign to variables
mrna_raw = data_files["mrna"]
meth_raw = data_files["meth"]
mirna_raw = data_files["mirna"]
protein_raw = data_files["protein"]
clinical_raw = data_files["clinical"]

print("\n✅ All files loaded successfully from GitHub!")

# ============================================================
# PART 1: DATA QUALITY ASSESSMENT
# ============================================================

print("\n" + "="*80)
print("PART 1: DATA QUALITY ASSESSMENT")
print("="*80)

# 1.1 TCGA HARMONIZATION
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
    
    axes[0, 0].hist(values, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f"{name} - Histogram")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    
    sns.kdeplot(values, fill=True, ax=axes[0, 1])
    axes[0, 1].set_title(f"{name} - KDE Plot")
    axes[0, 1].set_xlabel("Value")
    
    axes[0, 2].hist(values, bins=50, density=True, alpha=0.5, edgecolor='black')
    sns.kdeplot(values, ax=axes[0, 2])
    axes[0, 2].set_title(f"{name} - Density Plot")
    axes[0, 2].set_xlabel("Value")
    axes[0, 2].set_ylabel("Density")
    
    stats.probplot(values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f"{name} - Q-Q Plot")
    
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
# PART 2: FEATURE-LEVEL ANALYSIS
# ============================================================

print("\n" + "="*80)
print("PART 2: FEATURE-LEVEL ANALYSIS")
print("="*80)

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
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title(f"{name} - Correlation Heatmap (Top 50 Features)")
    plt.tight_layout()
    plt.show()
    
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
# PART 3: BIVARIATE ANALYSIS
# ============================================================

print("\n" + "="*80)
print("PART 3: BIVARIATE & BIOLOGICAL ANALYSIS")
print("="*80)

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
# 3.3 AGE VS OMICS
# ============================================================

print("\n" + "="*60)
print("3.3 AGE VS OMICS")
print("="*60)

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

# ============================================================
# 3.4 PAM50 VS OMICS (ANOVA)
# ============================================================

print("\n" + "="*60)
print("3.4 PAM50 VS OMICS (ANOVA)")
print("="*60)

if "PAM50 mRNA" in clinical_matched.columns:
    pam50_anova_results = []
    
    for name, df in datasets.items():
        top_features = df.var().sort_values(ascending=False).head(1000).index
        results = []
        subtypes = y.unique()
        
        for feature in top_features:
            try:
                groups = [df.loc[y == subtype, feature] for subtype in subtypes]
                f_stat, p_val = f_oneway(*groups)
                results.append([feature, f_stat, p_val])
            except:
                continue
        
        results_df = pd.DataFrame(results, columns=["Feature", "FStatistic", "PValue"])
        results_df = results_df.sort_values("PValue")
        
        rejected, p_adjusted, _, _ = multipletests(results_df['PValue'], method='fdr_bh')
        results_df['PValue_Adjusted'] = p_adjusted
        results_df['Significant'] = rejected
        
        print(f"\n📊 {name} ANOVA Results:")
        print(f"  Features tested: {len(results_df)}")
        print(f"  Significant features (FDR < 0.05): {rejected.sum()}")
        
        # Boxplots of top features
        top_features_plot = results_df.head(5)['Feature'].values
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features_plot):
            if i < len(axes):
                sns.boxplot(x=y, y=df[feature], ax=axes[i])
                axes[i].set_title(feature[:30])
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f"{name} - PAM50 vs Top 5 Features", fontsize=16)
        plt.tight_layout()
        plt.show()

# ============================================================
# PART 4: MULTI-OMICS INTEGRATION & GRAPH
# ============================================================

print("\n" + "="*80)
print("PART 4: MULTI-OMICS INTEGRATION & GRAPH")
print("="*80)

# 4.1 FEATURE FILTERING
# ============================================================

print("\n" + "="*60)
print("4.1 FEATURE FILTERING")
print("="*60)

def select_top_features(df, name, n_features):
    if n_features >= df.shape[1]:
        return df
    variances = df.var().sort_values(ascending=False)
    return df[variances.head(n_features).index]

top_mrna = select_top_features(mrna, "mRNA", 2000).index
top_meth = select_top_features(meth, "Methylation", 2000).index

print(f"mRNA: {mrna.shape[1]} -> 2000 features")
print(f"Methylation: {meth.shape[1]} -> 2000 features")
print(f"miRNA: {mirna.shape[1]} -> all features")
print(f"Protein: {protein.shape[1]} -> all features")

# ============================================================
# 4.2 INTEGRATED MATRIX
# ============================================================

print("\n" + "="*60)
print("4.2 INTEGRATED MATRIX")
print("="*60)

integrated = pd.concat([
    mrna[top_mrna].add_prefix("GE_"),
    meth[top_meth].add_prefix("METH_"),
    mirna.add_prefix("MIRNA_"),
    protein.add_prefix("PROT_")
], axis=1)

integrated = integrated.fillna(integrated.median())
print(f"Integrated Shape: {integrated.shape}")

# ============================================================
# 4.3 PCA WITH 95% VARIANCE
# ============================================================

print("\n" + "="*60)
print("4.3 PCA ANALYSIS")
print("="*60)

scaler = StandardScaler()
X = scaler.fit_transform(integrated)

pca_full = PCA()
pca_full.fit(X)

explained = pca_full.explained_variance_ratio_
cum_var = np.cumsum(explained)

n_components_95 = np.argmax(cum_var >= 0.95) + 1
print(f"Components to preserve 95% variance: {n_components_95}")

pca = PCA(n_components=n_components_95)
X_pca_95 = pca.fit_transform(X)
print(f"Transformed shape: {X_pca_95.shape}")

plt.figure(figsize=(10, 6))
plt.plot(cum_var[:100], linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'n={n_components_95}')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("PCA Variance Explained")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# PCA 2D
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X)

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
# 4.4 t-SNE & UMAP
# ============================================================

print("\n" + "="*60)
print("4.4 t-SNE & UMAP")
print("="*60)

tsne = TSNE(n_components=2, perplexity=min(30, len(y)-1), random_state=42, learning_rate='auto')
X_tsne = tsne.fit_transform(X_pca_95)

plt.figure(figsize=(10, 8))
for subtype in y.unique():
    mask = y == subtype
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=subtype, alpha=0.7)
plt.title("t-SNE Visualization")
plt.legend()
plt.tight_layout()
plt.show()

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
X_umap = reducer.fit_transform(X_pca_95)

plt.figure(figsize=(10, 8))
for subtype in y.unique():
    mask = y == subtype
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1], label=subtype, alpha=0.7)
plt.title("UMAP Visualization")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 4.5 kNN GRAPH
# ============================================================

print("\n" + "="*60)
print("4.5 kNN GRAPH")
print("="*60)

n_patients = X_pca_95.shape[0]
optimal_k = min(10, int(np.sqrt(n_patients)))
print(f"Optimal k: {optimal_k}")

A = kneighbors_graph(X_pca_95, n_neighbors=optimal_k, mode="connectivity", include_self=False)
G = nx.from_scipy_sparse_array(A)

components = list(nx.connected_components(G))
largest_comp = max(components, key=len)

graph_stats = pd.DataFrame({
    "Nodes": [G.number_of_nodes()],
    "Edges": [G.number_of_edges()],
    "Density": [nx.density(G)],
    "ConnectedComponents": [nx.number_connected_components(G)],
    "LargestComponent": [len(largest_comp)],
    "AverageDegree": [np.mean([d for _, d in G.degree()])],
    "AverageClustering": [nx.average_clustering(G)]
})

print("\n📊 GRAPH STATISTICS:")
print(graph_stats)

# Graph visualization
G_largest = G.subgraph(largest_comp)
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G_largest, seed=42, k=0.3)

color_palette = {
    'Basal-like': '#FF6B6B', 'HER2-enriched': '#4ECDC4',
    'Luminal A': '#45B7D1', 'Luminal B': '#96CEB4', 'Normal-like': '#FFEAA7'
}

node_indices = list(G_largest.nodes())
node_colors = []

for node_idx in node_indices:
    try:
        patient_id = integrated.index[node_idx]
        subtype = y.loc[patient_id]
        node_colors.append(color_palette.get(subtype, '#808080'))
    except:
        node_colors.append('#808080')

nx.draw(G_largest, pos, node_size=50, node_color=node_colors, with_labels=False, edge_color='gray', alpha=0.8)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_palette[subtype], label=subtype) for subtype in y.unique() if subtype in color_palette]
plt.legend(handles=legend_elements, loc='upper right')
plt.title(f"Patient Similarity Graph (Largest Component, n={len(largest_comp)})")
plt.tight_layout()
plt.show()

# ============================================================
# 4.6 GCN/GAT READY EXPORTS
# ============================================================

print("\n" + "="*60)
print("4.6 GCN/GAT READY EXPORTS")
print("="*60)

np.save("X_features.npy", X_pca_95)
np.save("adjacency.npy", A.toarray())

rows, cols = A.nonzero()
edge_list = pd.DataFrame({"Source": rows, "Target": cols})
edge_list.to_csv("edge_index.csv", index=False)

labels_df = pd.DataFrame({
    "Patient": integrated.index,
    "Subtype": y.values,
    "Subtype_Encoded": pd.Categorical(y).codes
})
labels_df.to_csv("labels.csv", index=False)

print(f"✅ Features shape: {X_pca_95.shape}")
print(f"✅ Adjacency edges: {len(edge_list)}")
print(f"✅ Labels: {len(y)} patients")

# ============================================================
# PART 5: BASELINE ML
# ============================================================

print("\n" + "="*80)
print("PART 5: BASELINE MACHINE LEARNING")
print("="*80)

# 5.1 LABEL ENCODING
# ============================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = encoder.classes_

print("\nClass Distribution:")
for i, name in enumerate(class_names):
    count = sum(y_encoded == i)
    percent = (count / len(y_encoded)) * 100
    print(f"  {i}: {name:15} {count:>3} patients ({percent:>5.1f}%)")

# ============================================================
# 5.2 TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    integrated, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

print(f"\nTrain Shape: {X_train.shape}")
print(f"Test Shape: {X_test.shape}")

# ============================================================
# 5.3 FEATURE SELECTION
# ============================================================

adaptive_k = min(1000, max(100, int(0.15 * integrated.shape[1])))
print(f"\nAdaptive Feature Selection: k={adaptive_k} features")

selector = SelectKBest(mutual_info_classif, k=adaptive_k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = integrated.columns[selector.get_support()]
print(f"Selected {len(selected_features)} features")

# ============================================================
# 5.4 MODELS
# ============================================================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
        eval_metric="mlogloss", num_class=len(class_names),
        random_state=42, use_label_encoder=False, verbosity=0
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", C=10, gamma='scale', probability=True,
        class_weight="balanced", random_state=42
    )
}

# ============================================================
# 5.5 MODEL TRAINING
# ============================================================

results = []
all_predictions = {}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print('='*60)
    
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"Accuracy:            {acc:.4f}")
    print(f"Balanced Accuracy:   {bal_acc:.4f}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"Weighted F1:         {weighted_f1:.4f}")
    print(f"MCC:                 {mcc:.4f}")
    
    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "BalancedAccuracy": bal_acc,
        "MacroF1": macro_f1,
        "WeightedF1": weighted_f1,
        "MCC": mcc
    })
    
    all_predictions[model_name] = y_pred
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ============================================================
# 5.6 MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(results_df.to_string(index=False))

# ============================================================
# 5.7 GNN READINESS REPORT
# ============================================================

print("\n" + "="*60)
print("GNN READINESS REPORT")
print("="*60)

best_idx = results_df['MacroF1'].idxmax()
best_model = results_df.loc[best_idx, 'Model']

gnn_report = pd.DataFrame({
    "Metric": [
        "Patients", "Integrated Features", "Selected Features",
        "Classes", "Graph Nodes", "Graph Edges", "Graph Density",
        "Best Model", "Best Accuracy", "Best Macro F1", "Best MCC"
    ],
    "Value": [
        integrated.shape[0], integrated.shape[1], adaptive_k,
        len(class_names), G.number_of_nodes(), G.number_of_edges(),
        f"{nx.density(G):.4f}",
        best_model,
        f"{results_df.loc[best_idx, 'Accuracy']:.4f}",
        f"{results_df.loc[best_idx, 'MacroF1']:.4f}",
        f"{results_df.loc[best_idx, 'MCC']:.4f}"
    ]
})

print(gnn_report.to_string(index=False))

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE - GNN READY")
print("="*80)

print("\n📊 EXECUTIVE SUMMARY:")
print("-" * 50)
print(f"Total Patients: {len(y)}")
print(f"Integrated Features: {integrated.shape[1]}")
print(f"Classes: {len(class_names)}")
print(f"Graph Nodes: {G.number_of_nodes()}")
print(f"Graph Edges: {G.number_of_edges()}")
print(f"Best Model: {best_model}")
print(f"Best Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
print(f"Best Macro F1: {results_df.loc[best_idx, 'MacroF1']:.4f}")

print("\n📁 Outputs Saved:")
print("  - X_features.npy")
print("  - adjacency.npy")
print("  - edge_index.csv")
print("  - labels.csv")
print("  - All visualizations displayed above")

print("\n" + "="*80)
print("🚀 RECOMMENDATION: Proceed to GNN/GAT Modeling")
print("="*80)