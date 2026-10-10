# AI-Based Student Performance Prediction and Study Recommendation System

## Overview
This beginner-friendly Streamlit application trains and compares a mean baseline, Linear Regression, and Random Forest on the UCI Student Performance dataset. It estimates the final grade (`G3`) from `G1`, `G2`, `studytime`, and `absences`, and provides simple rule-based study suggestions.

## Requirements
- Python 3.10 or newer recommended
- Internet connection on first run to fetch the dataset through `ucimlrepo`

## Run in VS Code (Windows)
1. Extract the project ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
5. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   python -m streamlit run app.py
   ```
7. The browser should open. Enter inputs and record the metrics shown by the app. Do not invent metrics for the PPT; copy the actual results from your run.

## What to explain during a demo
- `G3` is the target variable (final grade out of 20).
- `G1`, `G2`, `studytime`, and `absences` are selected input features.
- A fixed 80/20 train-test split is used.
- MAE and RMSE are errors in grade points; lower is better.
- R² indicates how much variation the model explains on this test split; it can be negative.
- The mean baseline is included so that ML models are not assumed to be better automatically.

## Important limitations
The dataset comes from two Portuguese secondary schools and may not generalize to Indian students. G1 and G2 are strongly correlated with G3, so this is a later-stage estimate, not an early-year forecast. Predictions are approximate and must not be used for high-stakes academic decisions.

## Data source
Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
Dataset page: https://archive.ics.uci.edu/dataset/320/student+performance
License listed by UCI: CC BY 4.0. Credit the source when sharing or adapting the data.
