# Notes on MAGNET Reproduction

## What was done

I cloned the official MAGNET code and ran the official inference script on BRCA, BLCA and OV datasets.

## Why this was done

Before modifying or improving MAGNET, it is necessary to reproduce the original paper results and check that the code and dataset setup are working correctly.

## Commands used

```bash
python main_inference.py --cfg configs/MAGNET_BRCA.yaml
python main_inference.py --cfg configs/MAGNET_BLCA.yaml
python main_inference.py --cfg configs/MAGNET_OV.yaml
```

## Final deliverable

The final deliverable is the 3-row reproduction table comparing my obtained accuracy with the paper accuracy.

## Next possible work

- weighted modality fusion
- modality contribution analysis
- masking versus imputation
- improved graph edge construction
