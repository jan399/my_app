#---Import of modules and functions---#

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualizer import (plot_confusion_matrix)

from utils.model_loader import (load_classification_report_L,
                                load_classification_report_S,
                                load_confusion_matrix_L,
                                load_confusion_matrix_S)

#---Start page ---#

st.set_page_config(page_title="Machine Learning")

st.markdown("""

# Machine Learning

During the machine learning phase, we explored three ensemble models.  
They combine multiple learners to create a stronger and more robust prediction:

- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
- [HistGradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)  
- [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)

After hyperparameter tuning using [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html),  
we selected **XGBoost** based on its performance and interpretability.

Let’s take a closer look.

### XGBoost Pipeline

This is the core code that powers our model.

Under the hood, it's doing quite a lot:
- Preprocessing the data  
- Running one of the most powerful ensemble algorithms in practice today  
- Handling multi-class predictions with carefully tuned parameters  

We chose XGBoost not just for its performance, but also for its robustness and interpretability.  
It's fast, flexible, and widely used in industry. It’s the engine behind our predictions in the RoleRecommender Application.
            
Curious? Just press the button below.
            """)

with st.expander(label="**Central Code for Machine Learning**"):

    st.markdown("""
    ```python
    # Determine the number of unique classes in each target set
    num_classes_S = len(np.unique(y_train_S))
    num_classes_L = len(np.unique(y_train_L))

    # Build pipelines, each with their own preprocessor

    # Pipeline for dataset S
    pipe_xgb_S = Pipeline([
        ('preprocessing', preprocessor_S),     # Uses the dedicated preprocessor for S
        ('xgb', XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes_S,           # Set according to unique classes in y_train_S
            eval_metric='mlogloss',
            random_state=42
        ))
    ])

    # Pipeline for dataset L
    pipe_xgb_L = Pipeline([
        ('preprocessing', preprocessor_L),     # Uses the dedicated preprocessor for L
        ('xgb', XGBClassifier(
            objective='multi:softmax',
            num_class=num_classes_L,           # Set according to unique classes in y_train_L
            eval_metric='mlogloss',
            random_state=42
        ))
    ])


    # Define the optimal hyperparameters found for Dataset S through Randomized Search
    best_params_S = {
        'xgb__reg_lambda': 1.0,         # L2 regularization term
        'xgb__n_estimators': 200,       # Number of boosting rounds (trees)
        'xgb__min_child_weight': 10,    # Minimum sum of instance weight (hessian) needed in a child
        'xgb__max_depth': 3,            # Maximum depth of a tree
        'xgb__learning_rate': 0.1       # Step size shrinkage to prevent overfitting
    }

    # Define the optimal hyperparameters found for Dataset L through Randomized Search
    best_params_L = {
        'xgb__reg_lambda': 2.0,
        'xgb__n_estimators': 300,
        'xgb__min_child_weight': 20,
        'xgb__max_depth': 6,
        'xgb__learning_rate': 0.1
    }

    # Set optimized hyperparameters (from Randomized Search or Grid Search) for each pipeline
    pipe_xgb_S.set_params(**best_params_S)
    pipe_xgb_L.set_params(**best_params_L)

    # Fit/train both pipelines with their respective training data
    pipe_xgb_S.fit(X_train_S, y_train_S)
    pipe_xgb_L.fit(X_train_L, y_train_L)

    # #---Predict and evaluate---#

    y_xgb_S = pipe_xgb_S.predict(X_test_S)
    y_xgb_L = pipe_xgb_L.predict(X_test_L) 
                
    ```

    """)

st.markdown("There’s more technical detail in our project documentation.  Let’s now focus on what matters most: **the model performance**.")

st.markdown("---")


# --- Radio selection for dataset ---
st.markdown("### Confusion Matrix")
st.markdown("""
**Confusion Matrix — What does it show?**
- Each **row** represents the **true class**
- Each **column** represents the **predicted class**
- The **diagonal** cells (top-left to bottom-right) show correct predictions
- **Off-diagonal** cells indicate misclassifications
- The **percentages** help identify how often each true class is confused with others
            """)

selection = st.radio(
    "#### Select a prediction model:",
    options=["**`Broad career role`**", "**`Specific career role`**"],
    horizontal=True
)

# --- Load and show the corresponding confusion matrix ---
try:
    if selection == "**`Broad career role`**":
        conf_dict = load_confusion_matrix_L()
        title = "Broad career role"
    else:
        conf_dict = load_confusion_matrix_S()
        title = "Specific career role"

    fig = plot_confusion_matrix(conf_dict, title=title)
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Error loading confusion matrix: {e}")

st.markdown("""
            
**What do these values tell us?**
- The **diagonal values** (e.g., "1252 (91.5%)" for Data Science in the broad role matrix, 
or "398 (75.4%)" for Data Scientist in the second matrix) show how often the model correctly predicts each class. 
High percentages here indicate strong model performance for these classes.
- The **off-diagonal values** reveal how often the model confuses different classes. For example, 
in the broad role matrix, "230 (54.9%)" in the lower left means that more than half of the true 'Tech' 
cases were wrongly predicted as 'Data Science'. This signals a challenge in distinguishing these roles.
- The **percentages** in each cell refer to the proportion of true samples in that row 
(i.e., how often a true class is assigned to each predicted label). This helps you quickly spot where the model tends to make mistakes and which classes are most/least often confused.
- In summary:  
    - A **strong diagonal** means the model is making accurate predictions for those roles.
    - **High off-diagonal values** highlight classes that are often confused—potentially showing where model 
    improvements or clearer feature engineering are needed.
---
""")

st.markdown("""
**What does this mean for the model's performance?**
- For **broad career roles** (Data Science vs. Tech):  
    - The model predicts 'DS' (Data Science) roles very accurately (over 91% correct).  
    - However, it struggles with 'Tech' roles—more than half (about 55%) of the true 'Tech' samples are misclassified as 'Data Science'.  
    - **Implication:** The model is biased towards predicting 'DS' and has difficulty distinguishing 'Tech' roles.

- For **specific career roles** (Data Analyst, Data Scientist, Software Engineer):  
    - The model predicts 'Data Scientist' and 'Software Engineer' reasonably well (around 70–75% correct), 
    but has problems with 'Data Analyst' (only ~47% correct).  
    - Many 'Data Analyst' roles are confused with 'Data Scientist', and some 'Software Engineer' cases 
    are predicted as 'Data Scientist'.
    - **Implication:** The model has trouble clearly separating 'Data Analyst' from 
    'Data Scientist' and occasionally confuses 'Software Engineer' with other data roles.

- **Overall:**  
    - The model shows good accuracy for some classes, but notable confusion for others.
    - The high off-diagonal values indicate specific areas where the model lacks discrimination between certain roles.
    - This suggests the need for better features, more balanced training data, or refined class definitions—especially 
    for separating 'Tech' from 'DS' and distinguishing 'Data Analyst' from 'Data Scientist'.
---
""")