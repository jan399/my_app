#---Import of modules and functions---#

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.statistic_functions import test_chisquare
from utils.data_loader import (
    load_df_heat_L,
    load_df_heat_S,
    load_question_long_short
)

from utils.visualizer import (plot_countplots, plot_heatmap_absolute, plot_heatmap_row_percent)

from utils.JanSimonLibrary import overview

#---Start page ---#

st.set_page_config(page_title="Data Analysis", page_icon="📊")

st.markdown("""
# Data Analysis and Visualization
### Two key decisions at the beginning

As part of the data exploration phase, we made two key decisions:

- **Define two sets of target variables:**  
  A career-oriented target set and a role-specific target set.  
  This approach allowed us to keep the model simple and transparent,  
  while covering both general career paths and specific job roles.  
  In short, we moved from a broad question:  
  *“What matters for a career path?”*  
  to a more focused one:  
  *“What matters for a specific role in Data Science?”*.

- **Reduce the number of features:**  
  We deliberately limited the number of input variables.  
  This involved dropping many questionnaire answers (columns)  
  to avoid overfitting and to improve clarity and interpretability.

Let’s take a closer look at how we defined our target variables.

---

### Target Variables

The target variable is based on **Question 5**:  
*“Select the title most similar to your current role (or most recent title if retired).”*

#### Broad career role

This groups several roles into broader career paths:

- **Data Science**  
  Includes the following titles:  
  - Data Analyst  
  - Data Scientist  
  - Machine Learning Engineer  
  - Data Engineer  
  - Research Scientist

- **Tech**  
  Includes the following titles:  
  - Software Engineer  
  - DBA / Database Engineer

#### Specific career role

In contrast to the broad roles, this version focuses on selected job titles without aggregation.  
We only included roles that are widely known and clearly defined:

- Data Analyst  
- Data Scientist  
- Software Engineer
""")


# --- 1. Let the user select the dataset ---
selection = st.radio(
    "#### Choose target type:",
    options=["**`Broad career role`**", "**`Specific career role`**"],
    index=0,
    horizontal=True
)

# --- 2. Load the corresponding DataFrame based on the user's selection ---
try:
    if selection == "**`Broad career role`**":
        df = load_df_heat_L()
    else:
        df = load_df_heat_S()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# --- 3. Perform analysis on the column at fixed index 2 ---
try:
    col_index = 2  # Fixed column index (third column, zero-based)
    col_name = df.columns[col_index]  # Get the column name for display
   

    # Generate the analysis text using the custom overview function
    analysis_text = overview(df, col_index)
    st.markdown(analysis_text)

except Exception as e:
    st.error(f"❌ Error during overview analysis: {e}")

st.markdown("""
---
""")


st.markdown("## Heatmap Visualization")

st.markdown("Let’s have a look how the carerr roles could be influenced by the answers from the questionnaire. What do you think is the most influential Faktor?")

# --- Step 1: Select the dataset via radio button ---
target_choice = st.radio(
    "Select target dataset:",
    options=["**`Broad career role`**", "**`Specific career role`**"],
    horizontal=True
)

# --- Step 2: Load the chosen dataframe ---
if target_choice == "**`Broad career role`**":
    df = load_df_heat_L()
else:
    df = load_df_heat_S()

# --- Step 3: Load mapping of long question texts to short labels ---
question_map_long_to_short = load_question_long_short()

st.write(f"Loaded DataFrame with shape {df.shape}")

# --- Step 4: Fix y-axis to column index 2 ---
y_index = 2
y_col_name = df.columns[y_index]

# --- Step 5: Prepare x-axis options (exclude y-axis column) ---
possible_x_cols = [col for col in df.columns if col != y_col_name]
x_display_names = [question_map_long_to_short.get(col, col) for col in possible_x_cols]

# --- Step 6: Select x-axis variable with display names ---
selected_display_name = st.selectbox(
    "Select x-axis variable:",
    x_display_names
)

# Map the selected display name back to the actual column name
selected_x_col = possible_x_cols[x_display_names.index(selected_display_name)]

# --- Step 7: Select value type (absolute or relative) ---
plot_type = st.radio(
    "Show values as:",
    options=["**`Absolute number per role`**", "**`Relative number per role`**"],
    index=0,
    horizontal=True
)

# --- Step 8: Plot heatmap with short label mapping ---
if plot_type == "**`Absolute number per role`**":
    plot_heatmap_absolute(df, selected_x_col, y_index=y_index, question_map=question_map_long_to_short)
else:
    plot_heatmap_row_percent(df, selected_x_col, y_index=y_index, question_map=question_map_long_to_short)


# Step 9: Perform Chi-Square test and display results
results = test_chisquare(df, df.columns.get_loc(selected_x_col), y_index)

st.markdown("""
            ---
            """)
st.markdown("### Chi-Square Test Results")
st.write(f"Chi2 Statistic: {results['Chi2 Statistic']:.4f}")
st.write(f"p-value: {results['p-value']:.4f}")
st.write(f"Degrees of Freedom: {results['Degrees of Freedom']}")
st.write(f"Cramér's V: {results['Cramers V']:.4f}")


st.markdown("""### What do these values tell us regarding correlations?

The **p-value** tells us how likely it is to observe this relationship *by chance alone*, assuming there's no real connection.  
If the p-value is below **0.05**, it’s considered statistically significant —  
in other words: it’s very unlikely this happened randomly.  
There’s probably some kind of meaningful relationship between the two variables.

**Cramér's V** goes one step further: it tells us *how strong* the relationship is.  
As a rule of thumb:

- **Weak association**: V ≤ 0.10  
- **Moderate association**: 0.10 < V ≤ 0.30  
- **Strong association**: V > 0.30  

In general, we can say: Yes, there is a correlartion between the answers and the roles. 
But it is weak when we consider each single answer. But what if we check for correlations of all answers in parallel? 
Or, in other words, **what if we apply a well selected Machine Learning algorithm for categorical features and targets?** 
You see, it’s getting a bit more technical now. We’re about to modelling!""")