# Machine Learning Assignment - 2  
**BITS Pilani — Work Integrated Learning Programme (WILP)**  
**Programme:** M.Tech (DSE)  
**Course:** Machine Learning

---

## a. Problem Statement

E-commerce platforms need to identify which browsing sessions are likely to end in a purchase.  
In this assignment, a binary classification pipeline is built on the **UCI Online Shoppers Purchasing Intention** dataset to predict session revenue outcome (`Purchase` vs `NoPurchase`).

The end-to-end workflow covers:

1. Dataset selection and preprocessing  
2. Training **five classical classification models** on the **same dataset**  
3. Computing Accuracy, AUC, Precision, Recall, F1, and MCC for every model  
4. Packaging trained artifacts in a GitHub repository  
5. Deploying an interactive **Streamlit** web application for test-data evaluation  

---

## b. Dataset Description

| Attribute | Details |
|---|---|
| **Dataset Name** | Online Shoppers Purchasing Intention Dataset |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| **Problem Type** | Binary Classification |
| **Target Variable** | `Revenue` → mapped to `Purchase` / `NoPurchase` |
| **Number of Instances** | **12,330** (≥ 500 required) |
| **Number of Features** | **17** (≥ 12 required) |
| **Class Balance** | Imbalanced (~84.5% NoPurchase, ~15.5% Purchase) |

### Feature Summary

**Numeric / continuous session metrics**

- `Administrative`, `Administrative_Duration`
- `Informational`, `Informational_Duration`
- `ProductRelated`, `ProductRelated_Duration`
- `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`

**Categorical / coded attributes**

- `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`

### Preprocessing & Experimental Setup

- Stratified train/test split: **80% / 20%** (`random_state=42`)  
  - Train rows: **9,864**  
  - Test rows: **2,466** (exported as `test_data.csv`)
- Numeric features: `StandardScaler`
- Categorical features: `OneHotEncoder(handle_unknown="ignore")`
- Class imbalance handled via `class_weight` where supported (Logistic Regression, Decision Tree, Random Forest)
- All models evaluated on the **same held-out test set**

---

## c. GitHub Repository Link

**Repository:** [https://github.com/thanuja8rr/online-shoppers-purchasing-intention](https://github.com/thanuja8rr/online-shoppers-purchasing-intention)

### Repository Structure

```text
ml-assignment-2/
├── app.py                      # Streamlit application
├── requirements.txt            # Deployment dependencies
├── README.md                   # This file
├── test_data.csv               # Held-out test data used in experiments
├── data/
│   └── online_shoppers_intention.csv
└── model/
    ├── train_models.py         # Training + evaluation script
    ├── preprocessor.joblib
    ├── label_encoder.joblib
    ├── feature_meta.json
    ├── metrics.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

### How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py          # optional re-train
streamlit run app.py
```

### Streamlit Community Cloud

**Live App:** [https://online-shoppers-purchasing-intention-ml-assignment.streamlit.app/](https://online-shoppers-purchasing-intention-ml-assignment.streamlit.app/)

---

## d. Models Used

The following classification models were implemented on the **same dataset** and evaluated with the **same metrics**:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier (kNN)  
4. Naive Bayes Classifier (GaussianNB)  
5. Ensemble Model — Random Forest  

### Comparison Table — Evaluation Metrics (Held-out Test Set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8500 | 0.8962 | 0.5107 | 0.7487 | 0.6072 | 0.5330 |
| Decision Tree | 0.8232 | 0.8513 | 0.4588 | 0.7880 | 0.5800 | 0.5057 |
| kNN | 0.8743 | 0.8018 | 0.6651 | 0.3796 | 0.4833 | 0.4391 |
| Naive Bayes | 0.6736 | 0.7932 | 0.2937 | 0.7880 | 0.4279 | 0.3234 |
| Random Forest (Ensemble) | **0.8933** | **0.9245** | **0.6476** | 0.6832 | **0.6650** | **0.6019** |


### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong linear baseline with high AUC (**0.8962**) and solid recall (**0.7487**). Precision is moderate because the positive (Purchase) class is minority; the model trades some false positives to catch more purchases. Overall, it is stable and interpretable. |
| **Decision Tree** | Highest recall among tree/linear models (**0.7880**) but lower precision (**0.4588**) and accuracy (**0.8232**). The single tree overfits local partitions of the imbalanced data, so generalization is weaker than the ensemble. |
| **kNN** | Highest accuracy among non-ensemble models (**0.8743**) with relatively good precision (**0.6651**), but **poor recall (0.3796)**. Distance-based voting under-detects the minority Purchase class in a high-dimensional one-hot feature space. |
| **Naive Bayes** | Weakest overall performer (Accuracy **0.6736**, MCC **0.3234**). GaussianNB assumes feature independence, which is unrealistic for correlated web-session metrics (`BounceRates`/`ExitRates`, duration features). High recall with very low precision indicates over-flagging of purchases. |
| **Random Forest (Ensemble)** | **Best model overall.** Highest Accuracy (**0.8933**), AUC (**0.9245**), F1 (**0.6650**), and MCC (**0.6019**), with a balanced precision–recall trade-off. Bagging + feature randomness handles non-linear interactions and class imbalance more effectively than individual classifiers. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — best discrimination (AUC) and best balanced correlation with true labels (MCC/F1) on the held-out test set. |

---

## Streamlit Application Features

The deployed app includes all mandatory features:

| Requirement | Implementation |
|---|---|
| Dataset upload (CSV test data) | Sidebar file uploader + bundled `test_data.csv` option |
| Model selection dropdown | All five trained models selectable |
| Display of evaluation metrics | Accuracy, AUC, Precision, Recall, F1, MCC |
| Confusion matrix / classification report | Side-by-side visualization and detailed report table |

---
