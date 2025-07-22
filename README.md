# RoleRecommender

RoleRecommender is a Streamlit app for recommending job roles and data science career paths based on user input.

## Demo

As soon as the app is deployed on Streamlit Cloud, you can try it out here:  
[RoleRecommender Online](https://share.streamlit.io/jan399/RoleRecommender-app/main/RoleRecommender-app.py)

*(Update the link after deployment)*

## Features

- Interactive user interface with Streamlit
- Recommendations for data science roles
- Visualizations with matplotlib and seaborn
- Data analysis with pandas and numpy
- Simple and intuitive operation

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/DEIN_USERNAME/RoleRecommender-app.git
    cd RoleRecommender-app
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the app:
    ```bash
    streamlit run RoleRecommender-app.py
    ```

## Directory structure
my_app
├── data
│   ├── categories.json
│   ├── classification_report_L.json
│   ├── classification_report_S.json
│   ├── confusion_matrix_L.json
│   ├── confusion_matrix_S.json
│   ├── default_X_train_L.csv
│   ├── default_X_train_S.csv
│   ├── desktop.ini
│   ├── df_heat_L.csv
│   ├── df_heat_S.csv
│   ├── df_long.csv
│   ├── encoder_assignment.csv
│   ├── Final Report Project Data Job.pdf
│   ├── kaggle_survey_2020_responses.csv
│   ├── pipe_xgb_L.pkl
│   ├── pipe_xgb_S.pkl
│   ├── Project Methodology.pdf
│   ├── Project Objective.pdf
│   ├── question_long_short.csv
│   ├── shap_feature_importance_all_classes_L.csv
│   ├── shap_feature_importance_all_classes_S.csv
│   ├── unique_values_per_feature_L.json
│   ├── unique_values_per_feature_S.json
│   └── unique_with_rank.csv
├── pages
│   ├── 01_ Introduction.py
│   ├── 02_Data Analysis and Visualization.py
│   ├── 03_Machine Learing.py
│   ├── 04_Your personal RoleRecommender.py
│   ├── 05_How this App was built.py
│   ├── 06_Download Center.py
│   └── desktop.ini
├── utils
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── data_loader.cpython-312.pyc
│   │   ├── desktop.ini
│   │   ├── JanSimonLibrary.cpython-312.pyc
│   │   ├── model_loader.cpython-312.pyc
│   │   ├── statistic_functions.cpython-312.pyc
│   │   └── visualizer.cpython-312.pyc
│   ├── __init__.py
│   ├── data_loader.py
│   ├── desktop.ini
│   ├── JanSimonLibrary.py
│   ├── model_loader.py
│   ├── statistic_functions.py
│   └── visualizer.py
├── .gitignore
├── desktop.ini
├── README.md
├── requirements.txt
├── RoleRecommender-app.py
└── terminal commands.txt
