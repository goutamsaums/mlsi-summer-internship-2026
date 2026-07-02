# MAGNET Reproduction Notes: BRCA, BLCA and OV

This note documents the reproduction of the official MAGNET model results on three datasets: **BRCA**, **BLCA**, and **OV**.

## Colab Notebook

The complete Google Colab notebook used for this experiment is available here:

[Open MAGNET Reproduction Colab Notebook](https://colab.research.google.com/drive/1MbfmIv8GrQzdNgpL9LkkTHMGU4-3ipBL?usp=sharing)


## Assignment

**Aim:**  
Download and run the official MAGNET code on all three datasets and check whether the reproduced accuracy values are close to the values reported in Table 1 of the MAGNET paper.

## Steps Followed

### Step 1: Clone and install MAGNET

The official MAGNET GitHub repository was cloned and the required Python dependencies were installed in Google Colab.

Official MAGNET repository:

`https://github.com/SinaTabakhi/MAGNET`

### Step 2: Prepare datasets

The preprocessed split datasets were used from:

`MAGNET_Datasets/split_data/`

The datasets used were:

- BRCA
- BLCA
- OV

Each dataset contains five split folders:

`10, 20, 30, 40, 50`

### Step 3: Run MAGNET on all three datasets

The official MAGNET inference script was run using the default dataset configuration files.

Commands used:

```bash
python main_inference.py --cfg configs/MAGNET_BRCA.yaml
python main_inference.py --cfg configs/MAGNET_BLCA.yaml
python main_inference.py --cfg configs/MAGNET_OV.yaml
```

### Step 4: Compare reproduced accuracy with paper accuracy

The reproduced accuracy values were compared with the accuracy values reported in Table 1 of the MAGNET paper.

The MAGNET paper reports accuracy as **mean ± standard deviation over five independent runs**.

## Reproduction Results

| Dataset | Previous Result | Final Result | Paper Accuracy | Paper Range | Roughly Matches? |
| ------- | --------------: | -----------: | -------------: | ----------: | ---------------- |
| BRCA    |          0.9198 | 0.9198 ± 0.0131 | 0.918 ± 0.012 | 0.906–0.930 | Yes |
| BLCA    |          0.9705 | 0.9705 ± 0.0062 | 0.970 ± 0.006 | 0.964–0.976 | Yes |
| OV      |          0.5753 | 0.5753 ± 0.0750 | 0.614 ± 0.052 | 0.562–0.666 | Yes |

## Matching Basis

The paper range was calculated as:

```text
Paper Range = Paper Mean ± Paper Standard Deviation
```

The reproduced mean accuracy values for **BRCA**, **BLCA**, and **OV** lie within the corresponding paper-reported ranges.

Therefore, all three reproduced results can be considered as roughly matching the MAGNET paper results.

## Output Files

| File / Folder | Description |
| ------------- | ----------- |
| `MAGNET_5run_reproduction_table.csv` | Final table with reproduced mean, standard deviation, paper range and matching result |
| `MAGNET_reproduction_table_clean.csv` | Earlier simple 3-row comparison table |
| `MAGNET_reproduction_table_detailed.csv` | Earlier detailed comparison table |
| `MAGNET_Reproduction_Accuracy_Report.pdf` | PDF comparison report containing the final table |
| `logs/` | Run logs for BRCA, BLCA and OV |
| `metric_summaries/` | Metric summaries extracted from MAGNET output files |

## Conclusion

The official MAGNET code was successfully executed on **BRCA**, **BLCA**, and **OV** datasets.

The reproduced mean accuracy values were:

- **BRCA:** 0.9198
- **BLCA:** 0.9705
- **OV:** 0.5753

These values fall within the corresponding paper-reported accuracy ranges. Hence, the reproduced results are consistent with the accuracy values reported in the MAGNET paper.

Therefore, the MAGNET setup, datasets, execution pipeline and final comparison table were successfully reproduced.
