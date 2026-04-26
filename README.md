# noisy-time-series-validation
A small Python project for cleaning, aligning, and validating noisy time-series measurements from two independent instruments.

# Noisy Time-Series Validation

This project demonstrates a simple Python workflow for validating noisy, stochastic time-series measurements from two independent instruments.

The workflow includes data loading, cleaning, resampling, timestamp matching, quality filtering, correlation analysis, bias calculation, RMSE calculation, and visualisation.

The data come from atmospheric turbulence monitoring, but the workflow is relevant to any setting where noisy signals need to be compared against an independent reference.

## What this project shows

- Python data analysis
- pandas time-series processing
- NumPy numerical analysis
- SciPy statistical metrics
- Matplotlib visualisation
- noisy real-world data cleaning
- timestamp alignment
- quality filtering
- correlation analysis
- bias and RMSE calculation

## Data

The script expects Polaris data in this Google Drive folder:

r0_values_log_2025_11_24.txt
/content/drive/MyDrive/Polaris_r0_data/
