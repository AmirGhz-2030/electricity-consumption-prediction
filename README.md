# ⚡ Smart Household Monthly Electricity Consumption Prediction with Machine Learning & Deep Neural Networks (MLP)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![README: English](https://img.shields.io/badge/README-English-brightgreen.svg)](README.md)
[![README: Persian](https://img.shields.io/badge/README-Persian-orange.svg)](README.fa.md)

> 🌐 **Read this in other languages:** **English** | [فارسی (Persian)](README.fa.md)

This project is an end-to-end machine learning system for accurately predicting monthly residential electricity consumption (in kWh). Using structured household and environmental features, three regression architectures are implemented and benchmarked: **Linear Regression** (baseline), **Multi-Layer Perceptron (`MLPRegressor`)**, and **XGBoost** ensemble trees. For practical real-world deployment, model inference is exposed through an interactive **Streamlit** web dashboard.

---

## ✨ Key Features

- **Data Leakage Protection Pipeline:** Standard Scikit-Learn preprocessing pipeline combining `SimpleImputer` for handling missing values and `StandardScaler` for feature normalization, ensuring reliable generalization on unseen test data.
- **Robust K-Fold Cross-Validation:** 5-fold cross-validation applied to the neural network confirming strong generalization without overfitting, achieving a remarkably low standard deviation (`0.025`).
- **Optimized Neural Network Architecture:** `MLPRegressor` configured with two hidden layers (128 and 64 neurons), ReLU activation, Adam optimizer, and Early Stopping.
- **Dual-Mode Interactive Dashboard:** Supports real-time single-household estimation via manual inputs as well as **Batch Inference** on bulk CSV files with formatted Excel export.
- **Built-in Local Datasets:** Core dataset files are bundled directly in the repository root for effortless academic reproducibility.

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning Frameworks:** Scikit-Learn, XGBoost
- **Data Processing & Numerics:** Pandas, NumPy
- **Web Dashboard & UI:** Streamlit
- **Interactive Charts & Export:** Plotly Express, OpenPyXL

---

## 📂 Dataset Overview

The datasets included in the repository root capture physical, environmental, and behavioral characteristics of residential units (sourced from a Quera data science competition):
- 📁 **`train.csv`:** 964 samples used for model training and parameter calibration.
- 📁 **`test.csv`:** 524 samples used for final evaluation and generalization testing.

### 📊 Input Features
1. `house_size_m2`: Floor area of the residence in square meters.
2. `num_residents`: Number of occupants living in the unit.
3. `avg_daily_temp`: Average daily ambient temperature (°C).
4. `heating_cooling`: On/off status of electric heating and cooling systems.
5. `high_power_appliances`: Count of active high-power electrical appliances.

- **Target Variable:** Monthly residential electricity consumption in kilowatt-hours (`electricity_consumption_kWh`).

---

## 📊 Model Evaluation & Benchmark Comparison

All three models comfortably surpassed the project target threshold ($R^2 > 0.60$). Final test-set $R^2$ scores are summarized below:

| Rank | Model Architecture | Accuracy Metric ($R^2$ Score) | Performance Summary |
| :---: | :--- | :---: | :---: |
| **1** | **Linear Regression** | **0.7470** | **Best Performance (Top Model)** |
| **2** | **Multi-Layer Perceptron (MLP Regressor)** | **0.7430** | **Exceptionally High Stability** |
| **3** | **Gradient Boosted Trees (XGBoost)** | **0.6730** | **Slight Overfitting** |

---

## 💻 Setup & Usage (Step-by-Step)

### 1. Install Dependencies
Clone the repository and install all required packages in one step:

```bash
pip install -r requirements.txt
```

### 2. Train Models & Export Artifacts
Open the project Jupyter notebooks (`bargh_LinearRegression.ipynb`, `bargh_mlp.ipynb`, and `bargh_xgboost.ipynb`) and run all cells to completion. Because the datasets live alongside the notebooks, file paths resolve automatically:
- Models are trained on the local `train.csv` dataset.
- Serialized model artifacts (`linear_model.joblib`, `mlp_model.joblib`, `xgboost_model.joblib`) and the data scaler (`scaler.joblib`) are generated in the project directory.

### 3. Launch the Streamlit Dashboard
Once the `.joblib` files are created, start the interactive dashboard from your terminal:

```bash
streamlit run app.py
```

---

## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
