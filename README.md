# Milestone2_team16
Milestone2_team16


# Movie Recommendation & Rating Prediction

This repository contains the code for Milestone 2 project of the MADS program.  
We implemented both Supervised Learning and Unsupervised Learning pipelines to build a movie rating prediction model and a recommendation system.

## Data Reproducibility Notice
```
To ensure reproducibility and consistent results in both unsupervised and supervised learning models, this project uses a frozen version of the final dataset: df_final_frozen_62188.csv.

Although the same codebase can generate a different number of records (e.g., 62,348 rows) due to updates in upstream sources (OMDB/MovieLens), we fix the dataset snapshot to 62,188 movies. This version has been validated and used throughout our modeling pipeline.

We strongly advise against regenerating the dataset from scratch, as it may cause inconsistencies in modeling results.

** Note **

The .ipynb files included in this repository are original and executable.
However, to meet repository size limitations, heavy files such as trained model pickle files (.pkl) and large datasets (e.g., rating.csv) are not included directly here.
Instead, we have:

Provided sample or placeholder files in appropriate directories to maintain reproducibility structure

Uploaded the complete datasets and models to Google Drive.

If you would like to download the full executable code, please click the link below.
[Download Link : https://drive.google.com/file/d/1WDLcvwp3oCh4jl5NvnRaF0RktxbLMTBv/view?usp=drive_link]

```

## Directory Structure
```
1. Data_preparation/
└── df_final_frozen_62188.csv

2. Supervised_learning_modeling/
├── supervised_pipeline.py
├── SL_1_SMOTE_4Model(RF, XGBoost, LogiReg, KNN).ipynb
├── SL_2_SMOTE_XGBoost_Hyper_tuning_4Model_Ensemble_Acc_0.6788_F1_0.6695.ipynb
├── SL_3_NoSMOTE_XGBoost_Hyper_tuning_4Model_Ensemble_Acc_0.6897_F1_0.6619.ipynb
└── SL_4_SMOTE_XGBoost(Hyper)_KNNRemoved_3Model_Ensemble_Acc_0.6922_F1_0.6677.ipynb

3. Unsupervised_learning_analysis/
├── UL_1_cluster_analysis_Kmeans_3d.ipynb
└── UL_2_cluster_analysis_DBSCAN_Failure_Case.ipynb

requirements.txt
README.md
```

## Features
```
- Text vectorization using `SentenceTransformer('all-mpnet-base-v2')`
- Multi-hot encoding of genres, actors, directors, etc.
- SMOTE for class imbalance handling
- Model evaluation with ROC, PR, and calibration curves
- DBSCAN clustering for unsupervised exploration
```

## Models Used for Supervised Learning
```
RandomForest
XGBoost (with tuning)
LogisticRegression
KNN (excluded in final version)
Model Ensemble (VotingClassifier)
```

## Models Used for Unsupervised Learning
```
KMeans Clustering
DBSCAN
```

## Visualization Outputs
```
This project includes:
Confusion matrix
Feature importance plots
ROC / PR / Calibration curves
Model performance comparison bar chart
```


## Reproducibility Notes
```
Python version: 3.9
Random seeds are fixed for consistent results
SMOTE-resampled data is cached (optional)
Trained models are saved in saved_models/ directory
```

## Requirements
```
Install dependencies:
pip install -r requirements.txt
```


## Team Member
- If you have any questions or get trouble, feel free to contact below team member!!
- Eric Kim : erikkim@umich.edu
- Shin choo : shinchoo@umich.edu
- Younghoon Oh : hooni@umich.edu
