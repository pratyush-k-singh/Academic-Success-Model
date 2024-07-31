import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Constants
DATA_PATH = 'data.csv'
OUTPUT_DIR = 'analysis_results'
TEST_SIZE = 0.2  # 20% of the data will be used for testing

def load_data():
    """Load data from CSV file."""
    data = pd.read_csv(DATA_PATH, delimiter=';')
    return data

def save_table(df, filename):
    """Save a DataFrame to a CSV file."""
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def save_plot(fig, filename):
    """Save a plot to a PNG file."""
    fig.savefig(os.path.join(OUTPUT_DIR, filename))

def analyze_data():
    """Perform data analysis and save results."""
    # Load data
    data = load_data()
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Drop 'id' column if exists
    if 'id' in train_data.columns:
        train_data.drop(columns='id', inplace=True)
    if 'id' in test_data.columns:
        test_data.drop(columns='id', inplace=True)
    
    # Check for null values and duplicates
    null_values_train = train_data.isnull().sum().sum()
    null_values_test = test_data.isnull().sum().sum()
    duplicates = train_data[train_data.duplicated(keep=False)]

    # Save info
    with open(os.path.join(OUTPUT_DIR, 'data_info.txt'), 'w') as f:
        f.write(f"Train data null value count: {null_values_train}\n")
        f.write(f"Test data null value count: {null_values_test}\n")
        f.write(f"Duplicate rows in train data: {len(duplicates)}\n")

    # Categorical and numerical features
    cat_features = [
        "Marital status", "Application mode", "Application order", "Course",
        "Daytime/evening attendance", "Previous qualification", "Nacionality",
        "Mother's qualification", "Father's qualification", "Mother's occupation",
        "Father's occupation", "Displaced", "Educational special needs", "Debtor",
        "Tuition fees up to date", "Gender", "Scholarship holder", "International"
    ]

    # Ensure only existing categorical features are used
    cat_features = [col for col in cat_features if col in train_data.columns]
    
    # Identify the target column if it exists
    target_column = 'Target' if 'Target' in train_data.columns else None

    # Exclude target column from numerical features if it exists
    num_features = list(train_data.drop(columns=cat_features + ([target_column] if target_column else [])).columns)
    
    # Save descriptive statistics
    save_table(train_data[num_features].describe(), 'numerical_description.csv')

    # Outlier analysis
    Q1 = train_data[num_features].quantile(0.25, numeric_only=True)
    Q3 = train_data[num_features].quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = ((train_data[num_features] < lower_bound) | (train_data[num_features] > upper_bound))
    outliers_count = outliers_iqr.sum()
    outliers_count = outliers_count[outliers_count > 0].sort_values(ascending=False)
    save_table(outliers_count, 'outliers_count.csv')
    
    # Correlation matrix plot
    if target_column:
        correlation_matrix = train_data[num_features + [target_column]].corr(numeric_only=True)
        if not correlation_matrix.empty:
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
            save_plot(plt, 'correlation_matrix.png')
            plt.close()
    
    # Target value counts plot
    if target_column:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=train_data[target_column].value_counts().index, y=train_data[target_column].value_counts().values, palette='crest')
        plt.ylabel('Count')
        plt.title('Target Value Counts')
        save_plot(plt, 'target_value_counts.png')
        plt.close()
    
    # Save training data features and target
    if target_column:
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        save_table(X_train, 'train_X.csv')
        save_table(y_train, 'train_y.csv')

        # Save testing data features and target
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        save_table(X_test, 'test_X.csv')
        save_table(y_test, 'test_y.csv')

if __name__ == "__main__":
    analyze_data()
