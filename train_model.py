import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_X_PATH = os.path.join(BASE_DIR, 'analysis_results', 'train_X.csv')
TRAIN_Y_PATH = os.path.join(BASE_DIR, 'analysis_results', 'train_y.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'academic_success_model.pkl')

def load_training_data():
    """Load training data."""
    X_train = pd.read_csv(TRAIN_X_PATH)
    y_train = pd.read_csv(TRAIN_Y_PATH).squeeze()  # Convert DataFrame to Series
    return X_train, y_train

def train_model():
    """Train the model and save it."""
    X_train, y_train = load_training_data()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    # Output training accuracy
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Training Accuracy: {train_accuracy:.2f}")
    
    # Detailed classification report
    print(classification_report(y_train, y_pred_train))

if __name__ == "__main__":
    train_model()
