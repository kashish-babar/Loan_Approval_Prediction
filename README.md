# Loan Approval Prediction - Machine Learning Model

This project aims to predict whether a loan application will be approved based on various features such as applicant’s credit history, marital status, education, loan amount, and other factors. The dataset consists of historical loan data, and the goal is to predict whether an applicant’s loan is approved or not.


## 🧰 Tools and Libraries Used

- **Python** – Programming language
- **Pandas** – Data manipulation and cleaning
- **Numpy** – Mathematical operations
- **Matplotlib** & **Seaborn** – Data visualization
- **Scikit-Learn** – Machine learning models and evaluation metrics
- **XGBoost** – Gradient boosting model


## 📝 Key Steps

1. **Data Cleaning**: Dropped the "Loan_ID" column, handled missing values by filling them with mode values, and performed label encoding on categorical features.
2. **Exploratory Data Analysis (EDA)**: Visualized the distribution of both categorical and numerical variables and explored correlations using heatmaps.
3. **Model Training**: 
   - **Logistic Regression**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
   - **XGBoost Classifier**
4. **Model Evaluation**: Each model was evaluated using accuracy metrics, and the best-performing model was selected.

## 📝 Model Comparison

| Model                  | Accuracy   |
|------------------------|------------|
| Decision Tree          | `0.7073` |
| Random Forest          | `0.7642` |
| XGBoost                | `0.7723` |
| Logistic Regression    | `0.7886` |

## 📈 Visualizations

- Distribution plots for categorical features.
- Histograms for numerical features.
- Correlation matrix heatmap.


## 📂 Repo Structure

- **Datasets**: Contains the dataset files for training and testing.
- **Main.py**: Python file for data analysis, preprocessing, and model training.
- **Requirements.txt**: List of dependencies to run the project.



---

  
Check out my GitHub: [https://github.com/kashish-babar]
