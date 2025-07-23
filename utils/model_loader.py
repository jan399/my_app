#---Import of modules and functions---#

import pandas as pd
import os
import joblib
import json
import pickle



# --- Load models using joblib ---

def load_model_L():
    """
    Load the XGBoost pipeline for general career path prediction (y_L).
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "data", "pipe_xgb_L.pkl")
    model = joblib.load(model_path)  # Load with joblib
    return model


def load_model_S():
    """
    Load the XGBoost pipeline for specific role prediction (y_S).
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "data", "pipe_xgb_S.pkl")
    model = joblib.load(model_path)  # Load with joblib
    return model


# Load report for measuring model performance
def load_classification_report_L(filename='classification_report_L.json'):
    """
    Load a classification report (JSON format) from the /data/ directory.
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    report_path = os.path.join(base_path, "data", filename)
    with open(report_path, "r") as f:
        return json.load(f)

def load_classification_report_S(filename='classification_report_S.json'):
    """
    Load a classification report (JSON format) from the /data/ directory.
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    report_path = os.path.join(base_path, "data", filename)
    with open(report_path, "r") as f:
        return json.load(f)
    
def load_confusion_matrix_L(filename='confusion_matrix_L.json'):
    """
    Load a classification report (JSON format) from the /data/ directory.
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    report_path = os.path.join(base_path, "data", filename)
    with open(report_path, "r") as f:
        return json.load(f)
    
def load_confusion_matrix_S(filename='confusion_matrix_S.json'):
    """
    Load a classification report (JSON format) from the /data/ directory.
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    report_path = os.path.join(base_path, "data", filename)
    with open(report_path, "r") as f:
        return json.load(f)



# --- Predict Probabilities ---
def predict_proba(model, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict class probabilities using the loaded pipeline.
    """
    probas = model.predict_proba(input_df)
    class_labels = model.classes_
    return pd.DataFrame(probas, columns=class_labels)