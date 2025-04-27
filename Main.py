import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# loading the dataset
train_df = pd.read_csv("Train_data.csv")
print(train_df.head())
test_df = pd.read_csv("Test_data.csv")
print(test_df.head())


# analyzing the data
print(train_df.info())
print(train_df.describe())
print(train_df.columns)
print(train_df.shape)


# load id is not needed , so drop it
train_df.drop(columns=["Loan_ID"], inplace=True)
test_df.drop(columns=["Loan_ID"], inplace=True)
print(train_df.shape)


# check for missing values
print(train_df.isnull().sum().sort_values(ascending=False))


# filling missing values
null_columns = [
    "Credit_History",
    "Self_Employed",
    "LoanAmount",
    "Dependents",
    "Loan_Amount_Term",
    "Gender",
    "Married",
]
for column in null_columns:
    if column in train_df.columns:
        train_df.fillna({column: train_df[column].mode()[0]}, inplace=True)
    if column in test_df.columns:
        test_df.fillna({column: test_df[column].mode()[0]}, inplace=True)

print(train_df.isnull().sum().sort_values(ascending=False))


# checking the list of all numeric and categorical columns
numeric = train_df.select_dtypes("number").columns.tolist()
print(numeric)
catagorical = train_df.select_dtypes("object").columns.tolist()
print(catagorical)


# checking the distribution of the target variable
for column in catagorical:
    print(train_df[column].value_counts(normalize=True))  # Shows % directly
    print("-" * 100)


# distribution of catagorical variables
for column in catagorical:
    plt.figure(figsize=(8, 10))
    sns.countplot(x=column, data=train_df, palette="Set2")
    plt.title(f"Distribution of {column}")
    plt.show()

# distribution of numeric variables
for column in numeric:
    plt.figure(figsize=(8, 10))
    sns.histplot(train_df[column], kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {column}")
    plt.show()


# encoding categorical variables
# (here, I'm facing an error, 'label_Status' , it is because traiin data have this column but test data doesn't)
le = LabelEncoder()
cat_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]  # Get all categorical columns except 'Loan_Status' for training and testing

for col in cat_cols:  # Encode each column
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

train_df["Loan_Status"] = le.fit_transform(
    train_df["Loan_Status"]
)  # Now separately encode 'Loan_Status' (ONLY in training set)


# plotting the correlation matrix
plt.figure(figsize=(10, 5))
sns.heatmap(train_df.corr(), cmap="cubehelix_r")
plt.xticks(rotation=20)
plt.title("Correlation Matrix")
plt.show()


# splitting the data into features and target variable
x = train_df.drop(columns=["Loan_Status"])
y = train_df["Loan_Status"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# model training with logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


# training the model with Decision Tree Classifier

DT_model = DecisionTreeClassifier(random_state=42)
DT_model.fit(X_train_scaled, y_train)

y_pred = DT_model.predict(X_test_scaled)
print("Predictions: ", y_pred)
# evaluate the model
accuracy = DT_model.score(X_test_scaled, y_test)
print("Accuracy: ", accuracy)


# training the model with Random Forest Classifier

RF_model = RandomForestClassifier(random_state=42)
RF_model.fit(X_train_scaled, y_train)

y_pred = RF_model.predict(X_test_scaled)
print("Predictions: ", y_pred)
# evaluate the model
accuracy = RF_model.score(X_test_scaled, y_test)
print("Accuracy: ", accuracy)


# model training with XGBoost

XG_model = XGBClassifier(random_state=42)
XG_model.fit(X_train_scaled, y_train)

y_pred = XG_model.predict(X_test_scaled)
print("Predictions: ", y_pred)
# evaluate the model
accuracy = XG_model.score(X_test_scaled, y_test)
print("Accuracy: ", accuracy)


# model training with logistic regression

LR_model = LogisticRegression(max_iter=2000, random_state=42)
LR_model.fit(X_train_scaled, y_train)

y_pred = LR_model.predict(X_test_scaled)
print("Predictions: ", y_pred)
# evaluate the model
accuracy = LR_model.score(X_test_scaled, y_test)
print("Accuracy: ", accuracy)
