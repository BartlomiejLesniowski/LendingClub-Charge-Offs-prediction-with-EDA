# LendingClub Charge-Offs Prediction with EDA

**Author:** Bartłomiej Leśniowski  
**Date:** 04.12.2024

A machine-learning pipeline to predict which LendingClub loans will charge off, accompanied by exploratory data analysis (EDA). The goal is to equip investors with an early-warning tool before a loan is fully funded.

---

## 📑 Table of Contents

1. [Project Overview] 
2. [Features & Dependencies]
3. [Installation]
4. [Dataset]
5. [Exploratory Data Analysis (EDA)] 
6. [Data Processing]
7. [Modeling]  
   - 7.1 [Train/Test Split]
   - 7.2 [Scaling & Encoding] 
   - 7.3 [Algorithms]
8. [Results & Evaluation]  
9. [Conclusions]
10. [Usage]

---

## Project Overview

LendingClub is the world’s largest P2P lending marketplace. In this project, we:

- Perform EDA on a ~1.55 GB loan dataset with 151 columns.  
- Engineer a target feature indicating “charged-off” vs. “fully paid.”  
- Train and compare four classifiers:  
  - Logistic Regression  
  - K-Nearest Neighbors  
  - Decision Tree  
  - Random Forest  
- Identify the best model for early risk detection.

---

## Features & Dependencies

- **Language:** Python 3  
- **Notebook:** Jupyter/IPython  
- **Key libraries:**  
  - pandas, numpy  
  - matplotlib, seaborn  
  - scipy  
  - scikit-learn (preprocessing, model_selection, metrics)  
  - cycler  

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/lendingclub-chargeoffs.git
   cd lendingclub-chargeoffs

2. **Create & activate virtual environment**

bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

3. **Install dependencies**

bash
pip install -r requirements.txt

**Dataset**

Source: LendingClub loan data (publicly available).

Size: ~1.55 GB, 151 columns

Selection criteria:

Only features known prior to loan funding

Dropped post-funding columns to prevent data leakage


**Exploratory Data Analysis (EDA)**

Checked distributions of key numeric features (e.g., loan amount, interest rate).

Assessed skew and applied log transforms where needed.

Visualized categorical counts (grade, sub_grade, purpose).

Correlation heatmaps to spot multicollinearity.


**Data Processing**

Missing values:

Dropped columns with > 90% missing.


Imputed median for numeric, mode for categorical.

Feature engineering:

Converted issue dates into year/month.

Binned continuous variables (e.g., annual income).


Target creation:

charge_off = 1 if status is “Charged Off,” else 0.



**Modeling**

Train/Test Split

Split data into 80% train / 20% test.

Stratified on target to preserve class balance.

Scaling & Encoding

StandardScaler on numeric features.

One-hot encoding for categoricals.


**Algorithms**

Model	- Key Params

Logistic Regression	- solver='lbfgs', max_iter=1000

K-Nearest Neighbors	- n_neighbors=5

Decision Tree	- max_depth=5

Random Forest	- n_estimators=100, max_depth=8


**Results & Evaluation**

Metrics used:

ROC-AUC

Precision, Recall, F1–score

Confusion Matrix


**Best performer: Random Forest**

ROC-AUC: 0.93

Precision/Recall/F1: 0.91 / 0.93 / 0.93


Detailed classification reports and ROC curves are in the notebook.


**Conclusions**

Random Forest and Decision Tree outperform simpler models in detecting charge-offs.

While not perfect, these models offer actionable signals to investors pre-funding.



**Usage**

Launch the notebook:

bash

jupyter notebook "LendingClub Charge Offs prediction with EDA.ipynb"

Follow each section sequentially to reproduce the analysis & modeling.


To retrain on new data, adjust the data-loading path and rerun cells in sections 2–9.
