# Dataset placement

Download `creditcard.csv` from the **Credit Card Fraud Detection** dataset by
the Machine Learning Group of Universite Libre de Bruxelles (ULB):

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file here:

```text
data/creditcard.csv
```

Expected columns:

- `Time`
- `V1` through `V28` (PCA-anonymized attributes)
- `Amount`
- `Class` (`0` = normal, `1` = fraud)

The notebook can also request OpenML dataset `1597`. The synthetic fallback is
strictly for code validation and its metrics must not be submitted as real-world
results. The CSV is intentionally excluded from Git because of size and dataset
distribution terms.

