import os
import streamlit as st

# Function for calling the file path
def get_file_path(filename):
    # Get the base directory of the current script (07_Download Center.py)
    # Then go up two levels to reach the 'my_app/' directory
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Join the base path with the 'data' directory and the given filename
    return os.path.join(base_path, "data", filename)

# Page Title
st.title("Download Center")

# --- Helper function to render download sections ---
def render_download_section(title, files_info):
    """
    Renders a section with a title and a list of downloadable files.
    """
    st.subheader(title)
    for file_info in files_info:
        filename = file_info["name"]
        mime_type = file_info["mime"]
        description = file_info["description"]
        
        file_path = get_file_path(filename)

        if os.path.exists(file_path):
            st.markdown(description)
            with open(file_path, "rb") as file:
                st.download_button(
                    label=f"Download {filename}",
                    data=file.read(),
                    file_name=filename,
                    mime=mime_type,
                    key=f"{title.replace(' ', '_')}_{filename}" # Unique key based on title and filename
                )
            st.markdown("---")
        else:
            st.warning(f"⚠️ File not found: **{filename}**")

# --- 1. Project Documents ---
project_documents = [
    {
        "name": "Project Objective.pdf",
        "mime": "application/pdf",
        "description": "**Project Assignment from Data Scientest**"
    },
    {
        "name": "Project Methodology.pdf",
        "mime": "application/pdf",
        "description": "**Project Methodology from Kaggle**"
    },
    {
        "name": "Final Report Project Data Job.pdf",
        "mime": "application/pdf",
        "description": "**Full Project Documentation from the project team (PDF Document)**"
    },
    {
        "name": "Questionaire.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "description": "**List of questions deroved from the raw data**"
    },
    {
        "name": "kaggle_survey_2020_responses.csv",
        "mime": "text/csv",
        "description": "**Raw data from Kaggle Survey 2020**"
    },
]
render_download_section("1. Project Documents", project_documents)

# --- 2. DataFrames for different purposes ---
dataframes_for_purposes = [
    {
        "name": "df_heat_L.csv",
        "mime": "text/csv",
        "description": "**Data for heatmap visualization (L)**"
    },
    {
        "name": "df_heat_S.csv",
        "mime": "text/csv",
        "description": "**Data for heatmap visualization (S)**"
    },
    {
        "name": "df_long.csv",
        "mime": "text/csv",
        "description": "**Long format of survey responses**"
    },
    {
        "name": "question_long_short.csv",
        "mime": "text/csv",
        "description": "**Mapping of long and short questions**"
    },
]
render_download_section("2. DataFrames for different purposes", dataframes_for_purposes)

# --- 3. Feature Preprocessing ---
feature_preprocessing_files = [
    {
        "name": "encoder_assignment.csv",
        "mime": "text/csv",
        "description": "**Encoder assignments for features**"
    },
    {
        "name": "unique_with_rank.csv",
        "mime": "text/csv",
        "description": "**Unique values with assigned ranks for ordinal encoding**"
    },
]
render_download_section("3. Feature Preprocessing", feature_preprocessing_files)

# --- 4. Machine Learning and Model Performance ---
ml_and_model_performance_files = [
    {
        "name": "classification_report_L.json",
        "mime": "application/json",
        "description": "**Classification report for model xgb_L**"
    },
    {
        "name": "classification_report_S.json",
        "mime": "application/json",
        "description": "**Classification report for for model xgb_S**"
    },
    {
        "name": "confusion_matrix_L.json",
        "mime": "application/json",
        "description": "**Confusion matrix for  model xgb_L**"
    },
    {
        "name": "confusion_matrix_S.json",
        "mime": "application/json",
        "description": "**Confusion matrix for  model xgb_S**"
    },
    {
        "name": "pipe_xgb_L.pkl",
        "mime": "application/octet-stream",
        "description": "**XGBoost pipeline for L**"
    },
    {
        "name": "pipe_xgb_S.pkl",
        "mime": "application/octet-stream",
        "description": "**XGBoost pipeline for S**"
    },
]
render_download_section("4. Machine Learning and Model Performance", ml_and_model_performance_files)

# --- 5. Feature Importance ---
feature_importance_files = [
    {
        "name": "shap_feature_importance_all_classes_L.csv",
        "mime": "text/csv",
        "description": "**SHAP feature importance for all classes (L)**"
    },
    {
        "name": "shap_feature_importance_all_classes_S.csv",
        "mime": "text/csv",
        "description": "**SHAP feature importance for all classes (S)**"
    },
    {
        "name": "unique_values_per_feature_L.json",
        "mime": "application/json",
        "description": "**Unique values per feature (L)**"
    },
    {
        "name": "unique_values_per_feature_S.json",
        "mime": "application/json",
        "description": "**Unique values per feature (S)**"
    },
]
render_download_section("5. Feature Importance", feature_importance_files)