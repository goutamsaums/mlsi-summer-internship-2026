# MAGNET Reproduction: BRCA, BLCA and OV

This folder contains reproduction results for the official MAGNET model on BRCA, BLCA and OV datasets.

## Aim

To run the official MAGNET code on all three datasets and check whether the obtained accuracy comes close to the values reported in the MAGNET paper.

## Dataset Source

The split datasets were copied from the official MAGNET repository and saved in this repository under:

`MAGNET_Datasets/split_data/`

## Paper Accuracy Values

| Dataset | Paper Accuracy |
|---|---:|
| BRCA | 0.918 |
| BLCA | 0.970 |
| OV | 0.614 |

## Final Reproduction Table

| Dataset | My Accuracy | Paper Accuracy | Roughly Matches? |
|---|---:|---:|---|
| BRCA | 0.9198 | 0.918 | Yes |
| BLCA | 0.9705 | 0.970 | Yes |
| OV | 0.5753 | 0.614 | Yes |

## Files

- `MAGNET_reproduction_table_clean.csv`: final 3-row deliverable table
- `MAGNET_reproduction_table_detailed.csv`: detailed table with standard deviation, difference and tolerance
- `logs/`: run logs for BRCA, BLCA and OV
- `metric_summaries/`: metric summaries extracted from MAGNET output files
