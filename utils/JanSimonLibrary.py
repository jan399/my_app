# We handle a big DF and expect to handle quite a lot of sub DF.
# Define a function "overview", that can be applied to a single DF.
# Attention: Cannot applied to a series or an array or multiple DFs.

def overview(dframe, col_index):
    col = dframe.columns[col_index]
    lines = []
    lines.append(f"**NAME COLUMN:** {col}")
    lines.append(f"- **Data Type:** {dframe[col].dtype}")
    lines.append(f"- **# Missing values:** {dframe[col].isna().sum()}")
    lines.append(f"- **Missing values Rate (%):** {round(dframe[col].isna().sum() * 100 / len(dframe), 2)}")
    
    lines.append(f"- **# Modalities:**\n\n```{dframe[col].value_counts().to_string()}```")
    return "\n".join(lines)