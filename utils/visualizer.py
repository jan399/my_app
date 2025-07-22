#---Import of modules and functions---#
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np


#---Simple Countplot to make a first dig into data---#

def plot_countplots(df: pd.DataFrame, columns: list):
    """
    Generates countplots for the given categorical columns using Seaborn and Matplotlib,
    applying custom CI colors.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): A list of column names for which to generate countplots.

    Returns:
        list: A list of Matplotlib Figure objects.
    """
    # Define CI colors
    PRIMARY = '#FF4B4B'     # Primary (Red)
    ACCENT = '#00A8E8'      # Accent (Blue)
    TEXT = '#31333F'        # Text (Dark gray)
    BACKGROUND = '#F0F2F6'  # Background (Light gray)

    figures = []  # This list will hold all the generated plot figures
    for col in columns:
        # Create a new Matplotlib figure and axes for each column
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(BACKGROUND)  # Set figure background color

        # Generate the countplot using the primary color for bars
        sns.countplot(
            x=df[col],
            ax=ax,
            color=PRIMARY
        )

        # Set the title with text color
        ax.set_title(f"Distribution: {col}\n", color=TEXT)
        # Rotate the X-axis labels and set their color
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, color=TEXT)
        # Set Y-axis tick color
        ax.set_yticklabels(ax.get_yticklabels(), color=TEXT)
        # Set axis labels color
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

        # Use tight layout for better spacing
        plt.tight_layout()

        # Add the generated figure to the list
        figures.append(fig)

        # Close the figure to free memory (important for Streamlit)
        plt.close(fig)

    return figures  # Return the list of figures, ready for st.pyplot()


#---Heatmap that is meant to fit ro a broad variety of features, targets and its categories---#
#---This is the heart of broad data visualization---#



def plot_heatmap_absolute(df: pd.DataFrame, x_col: str, y_index: int = 2, question_map=None):
    """
    Create and display a heatmap with absolute frequency counts,
    using CI colors for labels and background.

    Parameters:
    - df: Input DataFrame
    - x_col: Column name for the x-axis
    - y_index: Index of the y-axis column (default: 2)
    - question_map: Optional dict to map column names to short labels
    """
    # Define CI colors
    PRIMARY = '#FF4B4B'     # Primary (Red)
    ACCENT = '#00A8E8'      # Accent (Blue)  # Not used in this plot
    TEXT = '#31333F'        # Text (Dark gray)
    BACKGROUND = '#F0F2F6'  # Background (Light gray)

    # Get the y-axis column name by index
    y_col = df.columns[y_index]

    # Compute the frequency table (absolute counts)
    heatmap_data = pd.crosstab(df[y_col], df[x_col])

    # Create the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BACKGROUND)  # Set background color

    # Plot the heatmap with annotations (absolute counts), using Reds colormap
    sns.heatmap(
        heatmap_data,
        annot=True,      # Show numbers in each cell
        fmt='d',         # Integer format for absolute counts
        cmap='Reds',     # Use red colormap for heatmap
        ax=ax
    )

    # Set axis labels using short labels if provided, else use original names
    xlabel = question_map.get(x_col, x_col) if question_map else x_col
    ylabel = question_map.get(y_col, y_col) if question_map else y_col

    # Set label and title colors
    ax.set_xlabel(xlabel, fontsize=14, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=14, color=TEXT)
    ax.set_title("Heatmap of absolute role numbers\n", fontsize=18, color=TEXT)

    # Set tick label colors
    plt.setp(ax.get_xticklabels(), color=TEXT, rotation=45)
    plt.setp(ax.get_yticklabels(), color=TEXT, rotation=0)

    plt.tight_layout()
    st.pyplot(fig)


def plot_heatmap_row_percent(df: pd.DataFrame, x_col: str, y_index: int = 2, question_map=None):
    """
    Create and display a heatmap with row-wise percentage values,
    using CI colors for labels and background.

    Parameters:
    - df: Input DataFrame
    - x_col: Column name for the x-axis
    - y_index: Index of the y-axis column (default: 2)
    - question_map: Optional dict to map column names to short labels
    """
    # Define CI colors
    PRIMARY = '#FF4B4B'     # Primary (Red)
    ACCENT = '#00A8E8'      # Accent (Blue)  # Not used here
    TEXT = '#31333F'        # Text (Dark gray)
    BACKGROUND = '#F0F2F6'  # Background (Light gray)

    # Get the y-axis column name by index
    y_col = df.columns[y_index]

    # Compute the frequency table (absolute counts)
    heatmap_data = pd.crosstab(df[y_col], df[x_col])

    # Compute row-wise percentages
    row_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    # Format annotation values as percent strings with one decimal
    annot = row_pct.applymap(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")

    # Create the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BACKGROUND)  # Set background color

    # Plot the heatmap with percentage annotations, using Reds colormap
    sns.heatmap(
        row_pct,
        annot=annot,   # Show percentages in each cell
        fmt='',        # No formatting needed; annot has strings
        cmap='Reds',   # Use red colormap for heatmap
        ax=ax
    )

    # Set axis labels using short labels if provided, else original names
    xlabel = question_map.get(x_col, x_col) if question_map else x_col
    ylabel = question_map.get(y_col, y_col) if question_map else y_col

    # Set label and title colors
    ax.set_xlabel(xlabel, fontsize=14, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=14, color=TEXT)
    ax.set_title("Heatmap of Role Percentages (%)\n", fontsize=18, color=TEXT)

    # Set tick label colors
    plt.setp(ax.get_xticklabels(), color=TEXT, rotation=45)
    plt.setp(ax.get_yticklabels(), color=TEXT, rotation=0)

    plt.tight_layout()
    st.pyplot(fig)


# --- Plotting a confusion matrix (very nice stuff) ---#

def plot_confusion_matrix(conf_matrix_dict, title="Confusion Matrix"):
    """
    Plots a confusion matrix with both raw counts and row-wise percentages,
    using CI colors for labels and background.
    """

    # Define CI colors
    PRIMARY = '#FF4B4B'     # Primary (Red)
    ACCENT = '#00A8E8'      # Accent (Blue)  # Not used here
    TEXT = '#31333F'        # Text (Dark gray)
    BACKGROUND = '#F0F2F6'  # Background (Light gray)

    labels = conf_matrix_dict["labels"]
    matrix = np.array(conf_matrix_dict["matrix"])
    
    # Calculate row-wise percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    percentages = np.round(matrix / row_sums * 100, decimals=1)
    
    # Prepare cell annotations (e.g., "45\n(90%)")
    annotations = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            perc = percentages[i, j]
            annotations[i, j] = f"{count}\n({perc}%)"

    # Create figure and set background color
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BACKGROUND)  # Set background color

    # Plot heatmap using 'Reds' colormap (matches primary color)
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Reds",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        linewidths=0.5,
        linecolor="gray"
    )

    # Set title and axis labels using CI text color
    ax.set_title(title, fontsize=12, color=TEXT)
    ax.set_xlabel("Predicted Label", fontsize=12, color=TEXT)
    ax.set_ylabel("True Label", fontsize=12, color=TEXT)

    # Set tick label colors
    plt.setp(ax.get_xticklabels(), color=TEXT, rotation=45)
    plt.setp(ax.get_yticklabels(), color=TEXT, rotation=0)

    plt.tight_layout()
    return fig



# --- Plotting the SAHP values (very tricky stuff) ---#


def plot_shap_feature_importance_bar(df: pd.DataFrame, category: str, n_features: int = 15):
    """
    Creates a horizontal bar plot for SHAP feature importance using Matplotlib,
    applying CI colors for bars, labels, and background.

    Args:
        df (pd.DataFrame): DataFrame containing SHAP values, with a 'Question' column
                           and columns for each category (e.g., 'Data Science', 'Tech').
                           The 'Question' column is expected to already contain short question texts.
        category (str): The name of the column in df representing the selected category
                        whose SHAP values are to be plotted (e.g., 'Data Analyst').
        n_features (int): The number of top features to display in the plot.
                          Defaults to 15.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object containing the plot.
    """
    # Define CI colors
    PRIMARY = '#FF4B4B'     # Primary (Red)
    ACCENT = '#00A8E8'      # Accent (Blue)  # Not used here
    TEXT = '#31333F'        # Text (Dark gray)
    BACKGROUND = '#F0F2F6'  # Background (Light gray)

    # Ensure the DataFrame is sorted by the selected category's SHAP values in descending order
    # to place the most important features first.
    plot_df = df[['Question', category]].sort_values(by=category, ascending=False).head(n_features)

    # Create the Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BACKGROUND)  # Set background color

    # Create the horizontal bar plot using Seaborn with CI primary color
    sns.barplot(
        x=category, 
        y='Question', 
        data=plot_df, 
        ax=ax, 
        color=PRIMARY   # All bars in primary color
    )

    # Set axis labels and title with CI text color
    ax.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=12, color=TEXT)
    ax.set_ylabel("Question / Feature", fontsize=12, color=TEXT)
    ax.set_title(f"Top {n_features} Feature Importance for {category}", fontsize=14, color=TEXT)

    # Set tick label colors
    plt.setp(ax.get_xticklabels(), color=TEXT)
    plt.setp(ax.get_yticklabels(), color=TEXT)

    plt.tight_layout()

    return fig


# --- Plotting the Role Score for the RoleRecommender ---#

def plot_role_score_benchmark_vs_user(benchmark_score, user_score, class_name="Role"):
    """
    Plots a simple barplot comparing the role score of benchmark and user choices.
    """
    # Define CI colors
    PRIMARY = '#FF4B4B'     # Primary (Red)
    ACCENT = '#00A8E8'      # Accent (Blue)
    TEXT = '#31333F'        # Text (Dark gray)
    BACKGROUND = '#F0F2F6'  # Background (Light gray)

    scores = [benchmark_score, user_score]
    labels = ['Benchmark', 'Your Choices']
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor(BACKGROUND)  # Set background color

    bars = ax.bar(labels, scores, color=[PRIMARY, ACCENT])  # Set bar colors

    # Set axis labels and title with text color
    ax.set_ylabel('Role Score', color=TEXT)
    ax.set_ylim(0, 100)
    ax.set_title(f"Role Score for '{class_name}'", color=TEXT)

    # Set tick label colors
    ax.tick_params(colors=TEXT)
    plt.setp(ax.get_xticklabels(), color=TEXT)
    plt.setp(ax.get_yticklabels(), color=TEXT)

    # Annotate the bars with the exact value
    ax.bar_label(bars, fmt='%.1f', color=TEXT, fontsize=11)

    plt.tight_layout()
    return fig


#---Reading the File Strucuture---#

def build_tree(path: str, prefix: str = "") -> str:
    """
    Recursively build a tree structure as text.
    Folders are listed first, followed by files, both sorted alphabetically.
    """
    # Get all entries in the directory
    entries = os.listdir(path)

    # Sort entries: folders first, then files, both alphabetically
    entries = sorted(entries, key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))

    tree_str = ""
    for i, entry in enumerate(entries):
        # Use ├── for intermediate entries and └── for the last entry
        connector = "├── " if i < len(entries) - 1 else "└── "
        entry_path = os.path.join(path, entry)
        tree_str += f"{prefix}{connector}{entry}\n"

        # If entry is a directory, recursively add its content
        if os.path.isdir(entry_path):
            new_prefix = prefix + ("│   " if i < len(entries) - 1 else "    ")
            tree_str += build_tree(entry_path, new_prefix)

    return tree_str


#---Display the File Struture---+

def build_tree(path, prefix=""):
    # Liste von Dateien und Ordnern, die ignoriert werden sollen:
    ignore_names = {"desktop.ini", ".DS_Store", "__pycache__"}
    ignore_dirs = {".git", "__pycache__"}

    entries = [e for e in os.listdir(path) if e not in ignore_names]
    entries.sort()
    tree_str = ""
    for i, name in enumerate(entries):
        full_path = os.path.join(path, name)
        is_last = (i == len(entries) - 1)
        branch = "└── " if is_last else "├── "
        if os.path.isdir(full_path):
            if name in ignore_dirs:
                continue
            tree_str += f"{prefix}{branch}{name}\n"
            extension = "    " if is_last else "│   "
            tree_str += build_tree(full_path, prefix + extension)
        else:
            tree_str += f"{prefix}{branch}{name}\n"
    return tree_str

def show_project_tree():
    # Passe ggf. die Basis-Pfad-Logik an deine Ordnerstruktur an!
    base_path = os.path.dirname(os.path.dirname(__file__))
    project_name = os.path.basename(base_path)

    tree_output = f"{project_name}\n" + build_tree(base_path)
    st.markdown(f"```plaintext\n{tree_output}```")