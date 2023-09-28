import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, recall_score, precision_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleAutoML:
    def __init__(self, data):
        self.data = data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}

    def load_data(self, target_column):
        # Assuming data is a DataFrame with features and target
        self.X = self.data.drop(target_column, axis=1)
        self.y = self.data[target_column]
    
    def preprocess_data(self):
        # Standardize numerical features
        numerical_features = self.X.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        self.X[numerical_features.columns] = scaler.fit_transform(self.X[numerical_features.columns])

 # Encode categorical features if they exist
    def encode_data(self):
        categorical_features = self.X.select_dtypes(exclude=[np.number])

        if not categorical_features.empty:
            encoder = LabelEncoder()
            self.X[categorical_features.columns] = encoder.fit_transform(self.X[categorical_features.columns])
        else:
            print("No categorical features found in the data.")

    # Encode target variable if it is categorical
    def encode_target(self):
        if self.y.dtype == 'object':
            encoder = LabelEncoder()
            self.y = encoder.fit_transform(self.y)
        else:
            print("Target variable is not categorical, no encoding needed.")

    def visualize_numerical(self):
        # Univariate analysis for numerical columns
        numerical_features = self.X.select_dtypes(include=[np.number])
        for column in numerical_features.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=self.data, x=column, kde=True)
            plt.title(f'Univariate Analysis for {column}')
            plt.show()

        # Bivariate analysis for numerical columns
        for column1 in numerical_features.columns:
            for column2 in numerical_features.columns:
                if column1 != column2:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=self.data, x=column1, y=column2)
                    plt.title(f'Bivariate Analysis for {column1} vs {column2}')
                    plt.show()

    def visualize_categorical(self):
        # Univariate analysis for categorical columns
        categorical_features = self.X.select_dtypes(exclude=[np.number])
        for column in categorical_features.columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.data, x=column)
            plt.title(f'Univariate Analysis for {column}')
            plt.show()

        # Bivariate analysis for categorical columns
        for column1 in categorical_features.columns:
            for column2 in categorical_features.columns:
                if column1 != column2:
                    plt.figure(figsize=(8, 6))
                    sns.countplot(data=self.data, x=column1, hue=column2)
                    plt.title(f'Bivariate Analysis for {column1} vs {column2}')
                    plt.show()


    def handle_missing_values(self):
        # Check if there are any missing values in the data
        if self.data.isnull().values.any():
            # Measure the percentage of missing values for each column
            missing_percentage = self.data.isnull().mean() * 100

            # Separate numerical and categorical columns
            numerical_columns = self.X.select_dtypes(include=[np.number]).columns
            categorical_columns = self.X.select_dtypes(exclude=[np.number]).columns

            # Handle missing values in numerical columns (fill with mean)
            self.X[numerical_columns] = self.X[numerical_columns].fillna(self.X[numerical_columns].mean())

            # Handle missing values in categorical columns (fill with mode)
            self.X[categorical_columns] = self.X[categorical_columns].fillna(self.X[categorical_columns].mode().iloc[0])
        else:
            print("No missing values found in the data.")

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

    def select_models(self, task):
        # Automatically select appropriate models based on the task (classification or regression)
        if task == 'classification':
            self.models = {
                'RandomForestClassifier': RandomForestClassifier(),
                'LogisticRegression': LogisticRegression(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'SupportVectorClassifier': SVC(),
                # Add more classification models here as needed
            }
        elif task == 'regression':
            self.models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                # Add more regression models here as needed
            }
        else:
            raise ValueError("Invalid task. Supported tasks: 'classification' or 'regression'")

    def train_models(self):
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        results = {}
        for model_name, model in self.models.items():
            if isinstance(model, (RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, SVC)):
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                results[model_name] = {
                    'Accuracy': accuracy,
                    'F1 Score': f1,
                    'Recall': recall,
                    'Precision': precision,
                }
            elif isinstance(model, (RandomForestRegressor, LinearRegression, GradientBoostingRegressor)):
                y_pred = model.predict(self.X_test)
                if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
                    mse = mean_squared_error(self.y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(self.y_test, y_pred)
                    r2 = r2_score(self.y_test, y_pred)
                    results[model_name] = {
                        'Mean Squared Error': mse,
                        'Root Mean Squared Error': rmse,
                        'Mean Absolute Error': mae,
                        'R2 Score': r2,
                    }
        return results

if __name__ == "__main__":
    # Load your data
    data = pd.read_csv('data/pima.csv', header=None)
    data.columns = ['pregnancies', 'glucose', 'bp', 'skin_thickness', 'insulin', 'bmi', 'pedigree', 'age', 'label']

    # Initialize the AutoML package
    automl = SimpleAutoML(data)

    # Load data and perform EDA
    automl.load_data('label')

    # Handle missing values
    automl.handle_missing_values()

    # Preprocess data
    automl.preprocess_data()

    # Encode categorical features
    automl.encode_data()

    # Encode target variable
    automl.encode_target()

    # Visualize numerical and categorical columns separately, now with bivariate analysis
    automl.visualize_numerical()
    automl.visualize_categorical()

    # Split data into training and testing sets
    automl.split_data()

    # Automatically select appropriate models based on the task (classification or regression)
    task = 'classification'  # Change to 'regression' if your task is regression
    automl.select_models(task)

    # Train the selected models
    automl.train_models()

    # Evaluate the models
    results = automl.evaluate_models()
    for model_name, scores in results.items():
        print(f"{model_name} Performance:")
        for metric, score in scores.items():
            print(f"{metric}: {score}")
