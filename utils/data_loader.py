import os                               # For filenames, etc.
import pandas as pd                     # For data manipulation
import numpy as np                      # For numerical operations
import json                             # Allows uploading json files

#---This is are the original data from Kaggle---#

def kaggle_survey(filename='kaggle_survey_2020_responses.csv'):
    base_path = os.path.dirname(os.path.dirname(__file__))  # geht aus /utils/ raus
    data_path = os.path.join(base_path, "data", filename)
    return pd.read_csv(data_path, sep=',',dtype = 'str')

#---This is the list of all Q&As from the Kaggle qustionnaire. Derived from the RoleRecommnder notebook (v0.61)---#

def load_questionnaire(filename='Questionaire.xlsx', sheet_name=0):
    base_path = os.path.dirname(os.path.dirname(__file__))  # geht aus /utils/ raus
    data_path = os.path.join(base_path, "data", filename)
    return pd.read_excel(data_path, sheet_name=sheet_name, index_col=0)


#---Creates a dicitonary 'Questiones long - Questions short' derive from in df_heat (see function below)---#

def load_question_long_short(filename='question_long_short.csv'):
    # Get base directory
    base_path = os.path.dirname(os.path.dirname(__file__))
    # Build path to the CSV file
    data_path = os.path.join(base_path, "data", filename)
    # Load CSV without header because row 0 = long texts, row 1 = short texts
    df = pd.read_csv(data_path, sep=';', header=None)
    # Extract long texts from first row
    long_texts = df.iloc[0].tolist()
    # Extract short texts from second row
    short_texts = df.iloc[1].tolist()
    # Create dictionary mapping long texts (keys) to short texts (values)
    return dict(zip(long_texts, short_texts))


#---This is the data basically with all data, but we made it handable (look at the notebook RoleRecommender, latest version)---#

def load_df_long(filename='df_long.csv'):
    base_path = os.path.dirname(os.path.dirname(__file__))  # geht aus /utils/ raus
    data_path = os.path.join(base_path, "data", filename)
    df_long = pd.read_csv(data_path, sep=';')
    df_long = df_long.iloc[:, 1:]  # Entfernt die erste Spalte (z. B. 'Unnamed: 0')
    return df_long

#---This is the data for the data visualization and machine learning training and test stet.-(L)--#

def load_df_heat_L(filename='df_heat_L.csv'):
    # Get base project directory (two levels up from this file)
    base_path = os.path.dirname(os.path.dirname(__file__))
    # Build full path to data file inside 'data/' folder
    data_path = os.path.join(base_path, "data", filename)
    # Load CSV with semicolon separator and return DataFrame
    return pd.read_csv(data_path, sep=';')

#---This is the data for the data visualization and machine learning training and test set.-(S)--#

def load_df_heat_S(filename='df_heat_S.csv'):
    # Same as above, but loads the "role-specific" dataset
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", filename)
    return pd.read_csv(data_path, sep=';')

#--- This csv-file contains the data of the SHAP values for y_L'.---#
def load_shap_feature_importance_all_classes_L(filename='shap_feature_importance_all_classes_L.csv'):
    base_path = os.path.dirname(os.path.dirname(__file__))  # geht aus /utils/ raus
    data_path = os.path.join(base_path, "data", filename)
    df = pd.read_csv(data_path, sep=';')     # Read the .csv
    # Rename the columns    
    df = df.rename(columns={df.columns[0]: 
                            'Question', 
                            df.columns[1]: 
                            'Data Science',
                            df.columns[2]:
                            'Tech'
                            
                                                        })
    return df

#--- This csv-file contains the data of the SHAP values for y_L'.---#
def load_shap_feature_importance_all_classes_S(filename='shap_feature_importance_all_classes_S.csv'):
    base_path = os.path.dirname(os.path.dirname(__file__))  # geht aus /utils/ raus
    data_path = os.path.join(base_path, "data", filename)
    df = pd.read_csv(data_path, sep=';')     # Read the .csv
    # Rename the columns    
    df = df.rename(columns={df.columns[0]: 
                            'Question', 
                            df.columns[1]: 
                            'Data Analyst',
                            df.columns[2]:
                            'Data Scientist',
                            df.columns[3]:
                            'Software Engineer'})
    return df

#--- This json-file contains the data of selection field for the features in (L).---#

def load_unique_values_per_feature_L(filename='unique_values_per_feature_L.json'):
    """
    Load the json file from the directory
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    report_path = os.path.join(base_path, "data", filename)
    with open(report_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

#--- This json-file contains the data of selection field for the features in (S).---#

def load_unique_values_per_feature_S(filename='unique_values_per_feature_S.json'):
    """
    Load the json file from the directory
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    report_path = os.path.join(base_path, "data", filename)
    with open(report_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)
    
#--- This csv-file the default values for each feature (L).---#
    
def load_default_X_train_L(filename='default_X_train_L.csv'):
    # Same as above, but loads the "role-specific" dataset
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", filename)
    return pd.read_csv(data_path, sep=';', encoding='utf-8-sig')

#--- This csv-file the default values for each feature (S).---#

def load_default_X_train_S(filename='default_X_train_S.csv'):
    # Same as above, but loads the "role-specific" dataset
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", filename)
    return pd.read_csv(data_path, sep=';', encoding='utf-8-sig')

def load_unique_with_rank(filename='unique_with_rank.csv'):
    # Same as above, but loads the "role-specific" dataset
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", filename)
    return pd.read_csv(data_path, sep=';')

# --- This is the path loader for the PNG "Structuring the App with Modules.png" --- #


def load_app_structure_image(filename="Structuring the App with Modules.png"):
    # Get base project directory (two levels up from this file)
    base_path = os.path.dirname(os.path.dirname(__file__))
    # Build full path to the image inside 'data/' folder
    image_path = os.path.join(base_path, "data", filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return image_path