# Level 2 – Task 1: Predictive Modeling (Regression)

## Internship
Codveda Technologies – Data Science Internship

## Objective
Build and evaluate regression models to predict a continuous target variable (house prices) using a company-provided dataset.

## Dataset
- File: `house Prediction Data Set.csv`
- Type: Numerical dataset (housing data)

## Tools & Libraries
- Python
- pandas
- scikit-learn

## Workflow
1. Loaded the dataset and assigned appropriate column names.
2. Split the data into training and testing sets.
3. Trained multiple regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
4. Evaluated models using Mean Squared Error (MSE) and R² score.
5. Compared model performance.

## Results
- Linear Regression: Baseline performance
- Decision Tree: Improved accuracy over linear regression
- Random Forest: Best performance with lowest MSE and highest R² score
- ## Level 2 – Task 2: Classification (Churn Prediction)

### Objective
Build and evaluate classification models to predict customer churn using a real-world dataset.

### Dataset
- File: `churn-bigml-80.csv`
- Target variable: `Churn` (True / False)

### Tools & Libraries
- Python
- pandas
- scikit-learn

### Workflow
1. Loaded the churn dataset and explored basic structure.
2. Converted categorical features using one-hot encoding.
3. Applied feature scaling where required.
4. Trained a Logistic Regression classifier.
5. Evaluated model using Accuracy, Precision, Recall, and Confusion Matrix.
6. Compared results with a Random Forest classifier.

### Results
- Logistic Regression Accuracy ≈ 85%
- Random Forest Accuracy ≈ 94%
- Random Forest performed better, especially in identifying churn customers.

### Conclusion
Tree-based models handled the imbalanced dataset better than Logistic Regression and achieved higher recall for churn prediction.

### Author
Mohammed Abid  
Data Science Intern – Codveda Technologies


## Conclusion
Random Forest Regression provided the most accurate predictions among the tested models, making it the best choice for this dataset.

## Author
Mohammed Abid
