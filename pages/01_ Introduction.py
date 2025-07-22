#---Import of modules and functions---#

import pandas as pd
import streamlit as st
from utils.data_loader import load_df_long
from utils.visualizer import plot_countplots
from utils.JanSimonLibrary import overview

#---Start page ---#

st.set_page_config(page_title="Einführung", page_icon="🚀")

st.markdown("""

# Introduction

### Project Mission

This project was driven by a simple question:
What defines the key roles in today’s Data Science landscape – and how can data
help us understand them?
Our mission was to explore the technical roles emerging in the data industry by analyzing the tasks,
tools, and skills associated with each position. Based on this analysis, we built a role recommender
system designed to help aspiring professionals find the position that best aligns with their current
skills and career goals.
            
You might wonder whether it really makes sense to ask data experts how to work with data. 
However, when you consider typical decision-making situations, you’ll see that questions such
as “What is a key success factor for a specific goal?” or “What should my next best action (NBA) be?"
are common across many fields. We are pleased to share our insights with you and hope they help you make
well-informed decisions in your own journey.

---

### The Dataset

We used a rich dataset based on responses from over 20,000 participants to a global survey
conducted by Kaggle. The survey was designed to provide one of the most comprehensive views of
the data science and machine learning industry to date.

You can explore the dataset here:
https://www.kaggle.com/c/kaggle-survey-2020/overview


### Working with the Questionnaire

Overall, there are more than 300 possible answers given from the participants. Have a look:

""")

#---Show the Questiionnaire in a table---#

try:
    df_long = load_df_long()
    st.markdown("##### Questions and Answers")
    # Nur die Spaltennamen als eigene Spalte "Questions"
    column_table = pd.DataFrame({
        'Questions': df_long.columns
    })
    st.dataframe(column_table, height=300, width=800)


    # User input: column index
    st.markdown("##### Select a question")
  # st.write(f"Loaded DataFrame with shape {df_long.shape}")
    selected_index = st.number_input(
        "**Enter the column index (from the table above)**:",
        min_value=0,
        max_value=len(df_long.columns)-1,
        step=1
    )
    selected_col = df_long.columns[selected_index]
    st.write(f"You selected: **{selected_col}**")

    # Generate and show countplot
    figs = plot_countplots(df_long, [selected_col])
    st.pyplot(figs[0])

except Exception as e:
    st.error(f"❌ Error loading or plotting: {e}")

# Show analysis with overview function
try:
    st.markdown("""
**If you are more interested in numbers, get a quick information here:**
""")
    analysis_text = overview(df_long, selected_index)
    st.markdown(analysis_text)
except Exception as e:
    st.error(f"❌ Error during overview analysis: {e}")


st.markdown("""
---

### Conclusions

The original questionnaire includes more than 300 possible answers. To make the data usable and the results meaningful,
we made several key choices:

- **Question 4** served as our target variable, representing the role a participant currently holds.
  It became the basis for the role prediction in our model.
- All other questions were treated as input features for the machine learning model.
- For multiple-choice questions (e.g., Question 7), we didn’t evaluate which specific options were selected.
  Instead, we focused on the number of responses as a proxy for experience breadth.
  For example, when asked which programming languages someone knows, we counted how many languages were selected.
- Student-specific questions (e.g., those that have a capital B at the end, like Q29B) were excluded,
  as we assumed that participants with real-world experience offer more reliable indicators for career modeling.
- To keep the model focused and avoid overfitting, we deliberately limited the number of features.
  Although feature importance techniques could have helped with dimensionality reduction, we chose a simpler, more transparent approach.

**Look on the next page (select in the sidebar) to see how we proceed.**

---
""")