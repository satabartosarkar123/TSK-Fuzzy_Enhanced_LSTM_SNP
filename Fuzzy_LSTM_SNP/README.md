# Fuzzy LSTM-SNP Research: Data Processing Details

This directory contains the implementation and experimental results for the Fuzzy LSTM-SNP models. The following data processing pipeline is consistently applied across all experiments and datasets:

### 1. Normalization
All time series data are normalized to the range **[-1, 1]** using local min-max scaling. This ensures compatibility with the membership functions and the activation ranges of the LSTM-SNP cells.

### 2. Sliding Window Approach
A sliding window approach with lag **k** (typically $k=1$ or $k=2$ depending on the model variant) is used to construct supervised input-output pairs. This preserves the local temporal dependencies required for time series forecasting.

### 3. Train-Test Split
- **Test Data**: The final **60 observations** of each dataset are reserved exclusively for testing.
- **Temporal Ordering**: No shuffling is applied during the split or training process to strictly preserve the temporal ordering of the observations.

### 4. Feature Engineering
No additional feature engineering, smoothing, or filtering is performed beyond the initial normalization and differencing (where applicable), ensuring that the models operate on the raw temporal dynamics of the signal.

---
**Datasets Used:**
- S&P 500 Index
- Dow Jones Industrial Index
- Monthly Lake Erie Levels
- Monthly Milk Production
