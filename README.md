# Simple Model to Estimate Stellar Masses in the Bright Galaxy Survey

This project provides a straightforward approach to estimating stellar masses for galaxies in the Bright Galaxy Survey (BGS) using both photometric and spectroscopic data. The models developed aim to improve the accuracy of stellar mass estimates.

The project employs measurements from the Dark Energy Spectroscopic Instrument (DESI) to develop and validate a model.

---

## Repository Structure

- **data/**: Directory for storing input data files and model outputs.
- **doc/**: Documentation related to data sources and model descriptions.
- **job_management/**: Scripts and logs for managing batch jobs and model runs.
- **plots/**: Generated plots and visualizations from model outputs.
- **src/**: Source code for the models.
  - **linear/**: Linear regression model scripts.
  - **random_forest/**: Random forest model scripts including optimization.

---

## Dependencies

Requires **Python 3** and the following main libraries:

- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `scipy`  
- `shap` (for SHAP value analysis)  
- `optuna` (for hyperparameter tuning)

---

## Data and documentation
- [DESI Data](https://data.desi.lbl.gov/public/)
- [DESI Data Documentation](https://data.desi.lbl.gov/doc/)
- [DESI Early Data Release (EDR)](https://arxiv.org/abs/2306.06308)
- [Bright Galaxy Survey (BGS) Information](https://arxiv.org/abs/2208.08512)
