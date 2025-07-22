import streamlit as st
from utils.data_loader import load_app_structure_image
from utils.visualizer import show_project_tree


st.set_page_config(page_title="WHow this App was built")

st.markdown("# How this App was bulit")

st.markdown("### Feeding the App with data")
st.markdown("The steps below are processed in the notebook *ReoleRecommender v1.0.ipynd*. Access is easy via GitHub or the Download Center.")


# --- Data witht sub-Sections ---
steps = {
    1: {
        "title": "Data Preparation: First Insights",
        "items": [
            ("1.1", "Import Raw Data", ""),
            ("1.2", "Merge Question id and question text and use it as a column header", ""),
            ("1.3", "Replace NaNs in Single Choice questions with *z_not_answered*", ""),
            ("1.4", "Handle Multiple Choice Questions as Boolean Data", ""),
            ("1.5", "Export DataFrame *df_long.csv*",
             "Page **Introduction**: Bar plots for first rough data insights on page Introduction")
        ]
    },
    2: {
        "title": "Advanced Data Preparation: Features and Targets",
        "items": [
            ("2.1", "Features", ""),
            ("2.1.1", "Decide on relevant questions. Skip irrelevant questions.", ""),
            ("2.1.2", "Sum up the number of multiple choices in a single column per Question", ""),
            ("2.1.3", "Merge single and multiple-choice questions into one data frame", ""),
            ("2.2", "Targets", ""),
            ("2.2.1", "Skip irrelevant categories (Students, ...)", ""),
            ("2.2.2", "Generate `y_L` for Broad Career Focus and `y_S` for Specific Career Focus", ""),
            ("2.3", "Export DataFrame *df_heat_L.csv* and *df_heat_S.csv*",
             "Page **Data Visualization and Analysis**: Plotting heatmaps")
        ]
    },
    3: {
        "title": "Data preprocessing",
        "items": [
            ("3.1", "Define encoder type per target/feature in *encoder_assignment.csv*", ""),
            ("3.2", "Apply label encoder to target `y_L` and `y_S`", ""),
            ("3.3", "Feature Preprocessing to features without intrinsic order: Apply OneHotEncoder Pipeline `ohe_transformer`", ""),
            ("3.4", "Define Preprocessing features with intrinsic order (Ordinal Encoding)", ""),
            ("3.4.1", "Define intrinsic order for each feature that is assigned for Ordinal Encoding in file *unique_with_rank.csv*",
             "Page **Your personal RoleRecommender**: Generating category order for expander on page"),
            ("3.4.2", "Generate dictionary 'categories' for the corresponding parameter in `ord_transformer` Pipeline", ""),
            ("3.5", "Apply `ColumnTransformer` method on `preprocessor_L` and `preprocessor_S` pipelines", "")
        ]
    },
    4: {
        "title": "Split test set and training set",
        "items": [
            ("4.1", "Apply `train_test_split`-method", ""),
            ("4.2", "Export *default_X_train_L.csv* and *default_X_train_S.csv*",
             "Page **Your personal RoleRecommender**: Generate default feature categories (mode) for RoleRecommender benchmark")
        ]
    },
    5: {
        "title": "Machine Learning",
        "items": [
            ("5.1", "Instantiate pipelines `pipe_xgb_L` for Broad Career Role and `pipe_xgb_S` for Specific Career Role with initial and hyper-tuned parameters. Model selection and hyperparameter tuning by `RandomizedGrid` method is documented in *RoleRecommender_models.ipynb*", ""),
            ("5.2", "Train the model by `pipe_xgb_L.fit(X_train_L, y_train_L)` and `pipe_xgb_S.fit(X_train_S, y_train_S)`", ""),
            ("5.3", "Predict `y_test_L` (Broad career role) and `y_test_S` (Specific Career role)", ""),
            ("5.4", "Export *pipe_xgb_L.pkl*, *pipe_xgb_S.pkl*, *confusion_matrix_L.json* and *confusion_matrix_S.json* ",
             "Page **Your personal RoleRecommender**:<br>- Predict default and individual score values in RoleRecommender App<br>- Evaluate Model Performance")
        ]
    },
    6: {
        "title": "Feature Importance",
        "items": [
            ("6.1", "Calculate DataFrame with SHAP Values", ""),
            ("6.2", "Export *shap_feature_importance_all_classes_L.csv* and *shap_feature_importance_all_classes_S.csv*",
             "Page **Your personal RoleRecommender**: List the Top n features depending on the target class names and the selected number of features")
        ]
    }
}

# --- Render-function as markdown table ---
def render_step_as_table(step_num, step_data):
    table = "| **Nr.** | **Notebook Output** | **App Input** |\n"
    table += "|---------|-----------------|-------------|\n"
    for num, desc, comment in step_data["items"]:
        comment_html = (
            f"<div style='background-color:#f2f2f2; color:#333333; padding:6px; border-radius:5px;'>{comment}</div>"
            if comment else ""
        )
        table += f"| {num} | {desc} | {comment_html} |\n"
    st.markdown(table, unsafe_allow_html=True)

# --- Layout ---
for step_num, step_data in steps.items():
    with st.expander(f"### **Step {step_num}: {step_data['title']}**"):
        render_step_as_table(step_num, step_data)

st.markdown("---")


st.markdown("### Structuring the App with Modules")

# Import and display png-file
show_project_tree()