import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_shap_feature_importance_all_classes_L,
    load_shap_feature_importance_all_classes_S,
    load_question_long_short,
    load_default_X_train_L,
    load_default_X_train_S,
    load_unique_with_rank
)
from utils.model_loader import load_model_L, load_model_S
from utils.visualizer import plot_shap_feature_importance_bar, plot_role_score_benchmark_vs_user


st.set_page_config(page_title="What matters for your Career?")
st.markdown("# Your Personal RoleRecommender")


# --- Friendly scenario-based introduction ---

st.markdown("""
**Thinking about your next career move?**  
Start by exploring which qualifications are typically in demand for your target role. Then, reflect on your own strengths—where do you stand, and how might your skills improve your chances for a specific position? Finally, ask yourself the key question: *What’s your next step?*

**Welcome to the RoleRecommender!**  
Here, you’ll find answers to exactly these questions. Whether you’re exploring a general career in Data Science or aiming for a specific role, the RoleRecommender gives you practical insights and recommendations tailored to your path.

Ready to get started?
""")


# --- 0. Load feature list and value order from unique_with_rank.csv ---
rank_df = load_unique_with_rank()
features_with_order = list(rank_df.columns)
feature_options_with_order = {
    feat: [v for v in rank_df[feat] if pd.notnull(v)]
    for feat in features_with_order
}

# --- 1. Career Focus Selection ---
st.markdown("#### Choose your Career Focus")
career_focus = st.radio(
    "Select your type of career focus:",
    options=["**`Broad career role`**", "**`Specific career role`**"],
    index=0,
    horizontal=True
)

# --- 2. Category/Role Selection and data/model loading ---
if career_focus == "**`Broad career role`**":
    shap_df = load_shap_feature_importance_all_classes_L()
    model = load_model_L()
    default_input = load_default_X_train_L()
    category_columns = ["Data Science", "Tech"]
    category_map = {
        "Data Science": "0",
        "Tech": "1"
    }
else:
    shap_df = load_shap_feature_importance_all_classes_S()
    model = load_model_S()
    default_input = load_default_X_train_S()
    category_columns = ["Data Analyst", "Data Scientist", "Software Engineer"]
    category_map = {
        "Data Analyst": "0",
        "Data Scientist": "1",
        "Software Engineer": "2"
    }

# Map long question texts to their short forms (if you use mapping)
question_map = load_question_long_short()
shap_df["Question"] = shap_df["Question"].map(question_map).fillna(shap_df["Question"])

# --- 3. User selects a target role/category ---
st.markdown("#### Select your specific Role or Category")
selected_category = st.radio("Select your role:",
    options=category_columns,
    index=0,
    horizontal=True
)

# --- 4. Filter SHAP DataFrame to features in unique_with_rank.csv, keep order ---
filtered_shap_df = shap_df[shap_df["Question"].isin(features_with_order)].copy()
filtered_shap_df["feature_order"] = filtered_shap_df["Question"].apply(lambda x: features_with_order.index(x))
filtered_shap_df = filtered_shap_df.sort_values("feature_order")

# --- 5. Top-N Feature selection ---
st.markdown("#### How many top sucess factors should be shown?")
max_n = min(10, filtered_shap_df.shape[0])
n_features = st.slider(
    "How many relevant success factors to display?",
    min_value=1,
    max_value=max_n,
    value=min(5, max_n),
    key="num_features_slider"
)

# --- 6. Select top-N features by SHAP value (but keep unique_with_rank order) ---
top_features = (
    filtered_shap_df.sort_values(by=selected_category, ascending=False)
    ["Question"].iloc[:n_features]
    .tolist()
)

st.markdown("### Your Top Sucsess Factors")
with st.expander(f"See the top {n_features} features and their impact"):
    fig = plot_shap_feature_importance_bar(filtered_shap_df, selected_category, n_features=n_features)
    st.pyplot(fig)
    st.markdown("> **A higher bar means the feature has a stronger impact on your predicted role!**")
    st.dataframe(
        filtered_shap_df[["Question", selected_category]]
        .sort_values(by=selected_category, ascending=False)
        .head(n_features)
        .rename(columns={selected_category: "SHAP Value"})
    )

# --- 7. User input for each feature, with values in correct (ordinal) order ---
st.markdown("### Plan your next steps and understand their impact on your career goals!")
user_inputs = {}

for feature in top_features:
    values = feature_options_with_order.get(feature, ["Unknown"])
    default_value = default_input.at[0, feature] if feature in default_input.columns else values[0]
    user_inputs[feature] = st.selectbox(
        feature,
        options=values,
        index=values.index(default_value) if default_value in values else 0,
        key=f"select_{feature}"
    )

st.markdown(
    "<small>Note: All other features will be set to default values from the training data.</small>",
    unsafe_allow_html=True
)

# --- 8. Prediction Section with Role Score & Barplot ---
if st.button("Show Career Orientation Probabilities"):
    # Prepare user input DataFrame for prediction
    input_df = default_input.copy()
    for feature, value in user_inputs.items():
        input_df.at[0, feature] = value

    # Predict probabilities for user input and benchmark profile
    user_proba = model.predict_proba(input_df)
    benchmark_proba = model.predict_proba(default_input)

    # Map selected category (UI) to model class label (internal, always as String!)
    if career_focus == "**`Broad career role`**":
        category_map = {
            "Data Science": "0",
            "Tech": "1"
        }
    else:
        category_map = {
            "Data Analyst": "0",
            "Data Scientist": "1",
            "Software Engineer": "2"
        }
    selected_model_class = category_map[selected_category]

    # ALWAYS compare as strings for robustness!
    model_classes_str = [str(c) for c in model.classes_]
    selected_idx = model_classes_str.index(str(selected_model_class))

    # Calculate Role Scores (percent)
    user_score = float(user_proba[0, selected_idx]) * 100
    benchmark_score = float(benchmark_proba[0, selected_idx]) * 100

    # Show Role Score Comparison Plot
    fig = plot_role_score_benchmark_vs_user(benchmark_score, user_score, class_name=selected_category)
    st.pyplot(fig)



    # Optionally show all feature values used for prediction
    with st.expander("Show all input values used for prediction"):
        st.dataframe(input_df.T, use_container_width=True)

st.markdown("---")



st.markdown("""
### What’s happening here?

Technically speaking, we combine two things:

1. **SHAP Analysis:**  
   We use [SHAP](https://datascientest.com/de/shapley-additive-explanations-shap-was-ist-das) to explain how each of your answers (features) influences your predicted fit for each role
   (like Data Science, Tech, etc.). SHAP shows how much each answer increases or decreases the likelihood for a specific role. This helps you see which factors are most important for your personal result.

2. **Model Probabilities:**  
            For your choices, we calculate the probability for each role using the machine learning model:  
     `user_proba = model.predict_proba(input_df)`
            
    As a benchmark, we also calculate the probability based on the most common answers:  
    `benchmark_proba = model.predict_proba(default_input)`

As a result, under **"Your Choices"** you see your personalized probabilities for each role—so you can compare your result with the benchmark.

---
""")
