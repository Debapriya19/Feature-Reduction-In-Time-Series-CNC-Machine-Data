# Feature Reduction in Time Series CNC Machine Data

This repository contains the code and implementation for a collaborative academic project developed as part of the **Autonomous Multisensor Systems Lab** at **Otto von Guericke University (OVGU)** in 2025.

The goal of the project was to reduce the dimensionality of high-frequency time-series data collected from CNC machines. We explored three approaches for feature reduction — **Principal Component Analysis (PCA)**, a **Dense Autoencoder**, and a **sequence-aware LSTM+CNN Autoencoder** — and evaluated how well the reduced features could be used to predict electrical current for the spindle and X/Y/Z axes.

The focus was on reducing model complexity and sensor input size while preserving predictive accuracy and interpretability, with deployment constraints in mind (e.g., edge computing, limited latency).

---

### 🧪 Project Background

- **Type:** Academic lab project  
- **Course:** Autonomous Multisensor Systems (AMS)  
- **Institute:** Institute for Intelligent Cooperating Systems, OVGU  
- **Date:** September 2025

---

### 👥 Team Contributions

This was a group project with roles as documented in the [final report](/p3-feature-reduction-in-time-series-main/Feature_Reduction_final_presentation.pdf)

:

- **Debapriya Roy**  
  - Data engineering: schema validation, safe chronological splits  
  - Reproducibility setup: artifact versioning, random seed control  
  - SHAP-based model interpretability and edge deployment considerations  
  - Supported result alignment and reporting

- **Shweta Bambal**  
  - Designed and trained both the Dense and LSTM+CNN Autoencoder architectures  
  - Led hyperparameter tuning and ablation studies  
  - Focused on convergence diagnostics and joint loss tuning

- **Jayanthee Siva Perumal**  
  - Developed the PCA baseline and analyzed principal components  
  - Built downstream LSTM regressors for evaluation  
  - Led interpretability work (PCA loadings, SHAP) and authored core parts of the report

The team collaborated on experiment design, result validation, and documentation.

---

### 📂 Notes

- A **smaller trial version of the dataset** is included in this repository to demonstrate the pipeline.  
  The full dataset used in the original project is not shared due to size and restrictions.
- For experimental results, evaluation metrics, and analysis, refer to the final report.
