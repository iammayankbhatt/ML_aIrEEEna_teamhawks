# IEEE SB GEHU – Machine Learning Challenge (Device Fault Detection)

#### Team: Agile_Hawks
#### Members: Mayank Bhatt, Ansh Karki

This repository contains our solution for the IEEE SB GEHU Machine Learning Challenge.
The task is to determine whether a device is Normal (0) or Faulty (1) using 47 numerical sensor features (F01–F47).

---
## Project Overview

This problem was treated as a binary classification task.
We experimented with feature engineering, gradient boosting models, and ensemble techniques to improve the model’s ability to detect faulty devices.

The final pipeline achieved:

- **OOF Accuracy: 0.9903**

- **OOF F1 Score: 0.9877**
---
## Methodology

Our approach consists of the following main steps.

### 1. Feature Engineering

Since the sensor features are anonymized, additional features were created to extract more information from the data. These include:

- Row-wise statistics (mean, standard deviation, min, max, skewness)

- Sensor group sums

- Interaction features between nearby sensors

### 2. Feature Selection using SHAP

A baseline XGBoost model was trained and SHAP feature importance was used to identify useful features.
Most engineered features had non-zero importance, so all features were retained.

### 3. Hyperparameter Tuning

Hyperparameters for XGBoost, CatBoost, and LightGBM were tuned using Optuna.

### 4. Model Training

The three models were trained using 5-fold Stratified Cross Validation.
Class imbalance was handled using scale_pos_weight / class weights.

### 5. Stacking

Predictions from the three models were combined using a Logistic Regression meta-model.

### 6. Calibration and Threshold Selection

Isotonic calibration was applied to improve probability estimates.
Different classification thresholds were tested on OOF predictions and the one giving the best F1-score was selected.

---
## Repository Structure
```
solution_notebook.ipynb   # Training pipeline and experiments
FINAL.csv                 # Final predictions for the test set
full_pipeline.pkl         # Saved trained model
requirements.txt          # Python dependencies
README.md                 # Project documentation
```
---

## Training Environment

Model training and experimentation were performed using Kaggle Notebooks with access to an NVIDIA T4 GPU. This helped speed up the training of gradient boosting models such as XGBoost, CatBoost, and LightGBM, especially during cross-validation and hyperparameter tuning.

---
## Dataset

The dataset was provided as part of the IEEE SB GEHU ML Challenge.
The dataset used in this project was uploaded to Kaggle to make GPU training easier.

Kaggle Dataset Link:
https://www.kaggle.com/datasets/mayankbhatt1369/mlarena

The dataset contains:

TRAIN.csv – training data with labels

TEST.csv – test data used for generating predictions

Each sample contains 47 numerical sensor features (F01–F47) representing device measurements.

---

## ⚙️ Setup Instructions

To replicate our training environment locally, you will need Python 3.8+ installed on your system. 

*(Note: For maximum efficiency and exact replication of our pipeline, we highly recommend executing the training code utilizing an NVIDIA T4 GPU via Kaggle Notebooks or Google Colab.)*

1. Clone this repository to your local machine:
 ```bash
   git clone [https://github.com/iammayankbhatt/ML_aIrEEEna_teamhawks.git](https://github.com/iammayankbhatt/ML_aIrEEEna_teamhawks.git)
   cd ML_aIrEEEna_teamhawks
   ```
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```
## 💻 Usage Instructions
### Option 1: Retrain the Pipeline
1. Download the TRAIN.csv and TEST.csv datasets provided by the competition organizers.

2. Place both data files in the root directory of this cloned repository.

3. Open solution_notebook.ipynb using Jupyter Notebook, JupyterLab, or VS Code.

4. Ensure the file paths point to your local files (e.g., train_path = 'TRAIN.csv').

5. Run all cells sequentially. The script will handle SHAP pruning, Optuna tuning, ensemble training, threshold optimization, and output a new FINAL.csv.

### Option 2: Instant Inference
Use the included full_pipeline.pkl with joblib to load the pre-trained Level-1 and Level-2 models instantly. This bypasses the training phase entirely to generate predictions on new data efficiently.

---

## License

This repository is released under the MIT License.
