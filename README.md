# IEEE SB GEHU - Machine Learning Challenge: Device Fault Detection

**Team: Agile_Hawks** **Members:** Mayank Bhatt & Ansh Karki  

This repository contains our team's highly optimized, original solution for the online qualifiers of the Machine Learning Challenge hosted by IEEE SB, GEHU. The objective is to determine the operational status of embedded devices (Normal: `0` vs. Faulty: `1`) based on 47 continuous numerical sensor features (`F01` - `F47`).

## 🚀 Project Description & Methodology

We treated this as a binary classification problem. To maximize generalization and eliminate noisy sensor data, we engineered a rigorous, multi-stage machine learning pipeline that leverages GPU acceleration and advanced ensemble techniques. 

Our pipeline architecture includes:

1. **Domain-Agnostic Feature Engineering:** Engineered ~50 synthetic features (row-wise statistics, polynomial interactions, and sensor groupings). We used **SHAP (SHapley Additive exPlanations)** on a baseline XGBoost model to automatically prune zero-importance features and reduce dimensionality.
2. **Bayesian Hyperparameter Tuning:** Utilized **Optuna** with 5-Fold Cross Validation to dynamically discover the optimal learning rates, tree depths, and regularization constraints for our models.
3. **GPU-Accelerated Diverse Ensemble:** Trained a robust Level-1 ensemble of **XGBoost, CatBoost, and LightGBM**. Explicit class imbalance handling (`scale_pos_weight` and balanced auto-weights) was applied to ensure high sensitivity to the minority fault class.
4. **Level-2 Stacking & Isotonic Calibration:** Extracted Out-Of-Fold (OOF) probabilities from our base models and fed them into a Logistic Regression meta-classifier. We then applied Isotonic Regression to ensure our predicted probabilities represented true fault likelihoods.
5. **Dynamic Decision Thresholding:** By iterating through classification thresholds on our calibrated probabilities, we mathematically determined that a decision boundary of **0.52** maximized our F1-Score.

## 📊 Model Performance Metrics

During our rigorous 5-Fold Stratified Cross-Validation on the `TRAIN.csv` dataset, our stacking ensemble achieved the following Out-Of-Fold metrics:

* **Overall Accuracy:** 0.9903
* **Optimized F1-Score:** 0.9877 (at 0.52 threshold)
* **Precision (Faulty Class):** 0.99
* **Recall (Faulty Class):** 0.98

## 📁 Repository Structure

* `solution_notebook.ipynb`: The heavily documented Jupyter/Kaggle notebook containing data ingestion, feature engineering, Optuna tuning, model training, and meta-model stacking.
* `FINAL.csv`: The final prediction file generated for the `TEST.csv` dataset, formatted strictly as `ID -> CLASS`.
* `full_pipeline.pkl`: The serialized, ultra-lightweight pre-trained meta-ensemble and threshold configuration for instant inference without retraining.
* `requirements.txt`: List of required Python packages for easy setup.
* `README.md`: Setup, usage, and project documentation.
* `LICENSE`: MIT License covering the codebase.

## ⚙️ Setup Instructions

To replicate our training environment locally, you will need Python 3.8+ installed on your system. 

*(Note: For maximum efficiency and exact replication of our pipeline, we highly recommend executing the training code utilizing an NVIDIA T4 GPU via Kaggle Notebooks or Google Colab.)*

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/iammayankbhatt/ML_aIrEEEna_teamhawks.git](https://github.com/iammayankbhatt/ML_aIrEEEna_teamhawks.git)
   cd ML_aIrEEEna_teamhawks
Install the required dependencies:

Bash
pip install -r requirements.txt
💻 Usage Instructions
Option 1: Retrain the Pipeline
Download the TRAIN.csv and TEST.csv datasets provided by the competition organizers.

Place both data files in the root directory of this cloned repository.

Open solution_notebook.ipynb using Jupyter Notebook, JupyterLab, or VS Code.

Ensure the file paths point to your local files (e.g., train_path = 'TRAIN.csv').

Run all cells sequentially. The script will handle SHAP pruning, Optuna tuning, ensemble training, threshold optimization, and output a new FINAL.csv.

Option 2: Instant Inference
Use the included full_pipeline.pkl with joblib to load the pre-trained Level-1 and Level-2 models instantly. This bypasses the training phase entirely to generate predictions on new data efficiently.

📜 Declaration & License
All code and methodologies within this repository represent the original work of Team Agile_Hawks (Mayank Bhatt & Ansh Karki). The dataset used in this project is strictly for educational purposes as stipulated by the IEEE SB GEHU ML Challenge organizers. No ownership of the data is declared or implied.

The code in this repository is open-sourced under the MIT License.
