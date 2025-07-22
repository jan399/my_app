import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def test_chisquare(dframe: pd.DataFrame, x: int, y: int) -> dict:
    """
    Perform Chi-Square test and calculate Cramér's V between two categorical variables.

    Parameters:
    - dframe: input DataFrame
    - x: integer index of x-axis column (abscissa)
    - y: integer index of y-axis column (ordinate)

    Returns:
    - dict with Chi2 statistic, p-value, degrees of freedom, Cramér's V, and contingency table
    """
    ordinate_col_name = dframe.columns[y]
    abscissa_col_name = dframe.columns[x]

    # Create contingency table between the two categorical columns
    contingency_table = pd.crosstab(dframe[ordinate_col_name], dframe[abscissa_col_name])

    # Perform Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Calculate Cramér's V effect size
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    return {
        "Chi2 Statistic": chi2,
        "p-value": p,
        "Degrees of Freedom": dof,
        "Cramers V": cramers_v,
        "Contingency Table": contingency_table
    }
