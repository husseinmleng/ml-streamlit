# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import argparse
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Defining the class
class EDA:
    def __init__(self, data):
        self.data = data

    # Function to check the data types of the columns
    def check_data_type(self):
        print("Data Types of the columns are: ")
        print(self.data.dtypes)
        print("\n")

    # Function to check the shape of the dataset
    def check_shape(self):
        print("Shape of the dataset is: ")
        print(self.data.shape)
        print("\n")

    # Function to check the missing values in the dataset
    def check_missing_values(self):
        print("Missing values in the dataset are: ")
        print(self.data.isnull().sum())
        print("\n")

    # Function to check the missing values in the dataset
    def check_missing_values_percentage(self):
        print("Missing values percentage in the dataset are: ")
        print(self.data.isnull().sum() / len(self.data) * 100)
        print("\n")

    # Function to visualize numerical columns
    def visualize_numerical_columns(self):
        print("Visualizing the numerical columns: ")
        numerical_columns = self.data.select_dtypes(include=np.number).columns.tolist()
        
        # Define the number of rows and columns for the grid
        num_rows = (len(numerical_columns) + 2) // 3
        num_cols = min(len(numerical_columns), 3)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        
        for i, column in enumerate(numerical_columns):
            row, col = divmod(i, num_cols)
            ax = axes[row, col]
            
            # Create a histogram for the numerical column
            sns.histplot(data=self.data, x=column, ax=ax, kde=True)
            ax.set_title(f"Histogram for {column}")
            ax.set_xlabel(column)
        
        # Remove any empty subplots
        for i in range(len(numerical_columns), num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.show()
        print("\n")

    
    # Function to visualize categorical columns
    def visualize_categorical_columns(self):
        print("Visualizing the categorical columns: ")
        categorical_columns = self.data.select_dtypes(exclude=np.number).columns.tolist()
        
        if not categorical_columns:
            print("No categorical columns to visualize.")
            return

        # Define the number of rows and columns for the grid
        num_rows = (len(categorical_columns) + 2) // 3
        num_cols = min(len(categorical_columns), 3)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        
        for i, column in enumerate(categorical_columns):
            row, col = divmod(i, num_cols)
            ax = axes[row, col]
            
            # Create a count plot for the categorical column
            sns.countplot(data=self.data, x=column, ax=ax)
            ax.set_title(f"Count Plot for {column}")
            ax.set_xlabel(column)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        # Remove any empty subplots
        for i in range(len(categorical_columns), num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.show()
        print("\n")
        
    # Function to visualize the correlation between numerical columns
    def visualize_correlation(self):
        print("Visualizing the correlation between the numerical columns: ")
        numerical_columns = self.data.select_dtypes(include=np.number).columns.tolist()
        sns.heatmap(self.data[numerical_columns].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
        print("\n")

    # Function to display additional information about the dataset
    def dataset_info(self):
        print("Additional Dataset Information:")
        print("Number of unique values in each column:")
        print(self.data.nunique())
        print("\n")

# define system arguments

parser = argparse.ArgumentParser()
parser.add_argument("--kind",  help="kind of data e.g. csv, excel, sql",required=True)
parser.add_argument("--path",  help="path of the data",required=False)
parser.add_argument("--table", help="name of the table in sql database",required=False)
parser.add_argument('--data_file', type=str, help='Path to the dataset file')
parser.add_argument('--models', default=['LogisticRegression'], 
                    help='List of models to train and evaluate (default: LogisticRegression)')
parser.add_argument('--metrics', default=['accuracy'], 
                    help='List of evaluation metrics to compute (default: accuracy)')


args = parser.parse_args()
# Load the dataset
if args.data_file.endswith(".csv"):
    data = pd.read_csv(args.data_file)
elif args.data_file.endswith(".xlsx"):
    data = pd.read_excel(args.data_file, sheet_name="Sheet1")
elif args.data_file.endswith(".db"):
    conn = sqlite3.connect(args.data_file)
    data = pd.read_sql_query("SELECT * FROM data_table", conn)
else:
    raise ValueError("Invalid data file format. Supported formats are: csv, xlsx, db")

# Initialize EDA instance
eda = EDA(data)
eda.check_data_type()
eda.check_shape()
eda.check_missing_values()
eda.check_missing_values_percentage()
eda.visualize_numerical_columns()
eda.visualize_categorical_columns()
eda.visualize_correlation()

# Split the dataset into features and target variable
X = data.drop(args.target, axis=1)
y = data[args.target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a dictionary to store model results
results = {}

# Train and evaluate selected models
for model_name in args.models:
    if model_name == 'LogisticRegression':
        model = LogisticRegression()
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model_name == 'RandomForest':
        model = RandomForestClassifier()
    else:
        raise ValueError(f"Invalid model choice: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {}
    for metric_name in args.metrics:
        if metric_name == 'accuracy':
            metric_value = accuracy_score(y_test, y_pred)
        elif metric_name == 'precision':
            metric_value = precision_score(y_test, y_pred)
        elif metric_name == 'recall':
            metric_value = recall_score(y_test, y_pred)
        elif metric_name == 'f1':
            metric_value = f1_score(y_test, y_pred)
        else:
            raise ValueError(f"Invalid metric choice: {metric_name}")

        metrics[metric_name] = metric_value

    results[model_name] = metrics

# Print the evaluation results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print()


