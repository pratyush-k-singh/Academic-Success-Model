import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_X_PATH = os.path.join(BASE_DIR, 'analysis_results', 'test_X.csv')
TEST_Y_PATH = os.path.join(BASE_DIR, 'analysis_results', 'test_y.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'academic_success_model.pkl')
OUTPUT_DIR = os.path.join(BASE_DIR, 'prediction_results')

def load_test_data():
    """Load test data."""
    X_test = pd.read_csv(TEST_X_PATH)
    y_test = pd.read_csv(TEST_Y_PATH).squeeze()  # Convert DataFrame to Series
    return X_test, y_test

def load_model():
    """Load the trained model."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def save_plot(fig, filename):
    """Save a plot to a PNG file."""
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')

def make_predictions():
    """Make predictions and save results."""
    X_test, y_test = load_test_data()
    model = load_model()
    
    y_pred = model.predict(X_test)
    
    # Save predictions
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
    
    # Output test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    # Detailed classification report
    print(classification_report(y_test, y_pred))
    
    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    save_plot(fig, 'confusion_matrix.png')

if __name__ == "__main__":
    make_predictions()
