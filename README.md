# Titanic Survival Prediction

## Project Overview

This project uses machine learning to predict whether Titanic passengers survived the shipwreck based on passenger information.

Built as a beginner machine-learning project to practise data analysis, data preprocessing, classification and model evaluation.

## Dataset

The dataset is from Kaggle's Titanic: Machine Learning from Disaster competition.

- Training data: 891 passengers
- Test data: 418 passengers
- Target variable: `Survived` (0 = did not survive, 1 = survived)

## Tools and Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Project Workflow

1. Explored the passenger dataset and missing values.
2. Analysed survival rates by gender, passenger class and age group.
3. Handled missing values and encoded gender.
4. Selected features: `Pclass`, `Sex`, `Age`, `Fare`, `SibSp` and `Parch`.
5. Split the labelled data into training and validation sets.
6. Trained Logistic Regression and Decision Tree classifiers.
7. Evaluated accuracy, precision, recall, F1-score and the confusion matrix.
8. Used five-fold cross-validation for Logistic Regression.
9. Generated survival predictions for the competition test dataset.

## Models

- Logistic Regression
- Decision Tree Classifier

The initial Logistic Regression model achieved approximately 80.45% accuracy on one held-out validation split. Results may vary depending on the data split and preprocessing.

## How to Run

1. Download the Titanic dataset from Kaggle.
2. Place the CSV files in the appropriate project folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Run the Python script:

   `python Tain.py`

## Output

The project generates `submission.csv` containing passenger IDs and predicted survival labels in Kaggle's required format.

## What I Learned

- Exploring and cleaning real-world data
- Handling missing values
- Feature selection and encoding
- Training classification models
- Evaluating models and interpreting confusion matrices
- Understanding train-test splits and cross-validation

## Dataset Source

Kaggle — Titanic: Machine Learning from Disaster  
https://www.kaggle.com/competitions/titanic
