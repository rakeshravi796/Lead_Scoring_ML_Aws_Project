import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LeadScoringMinimalCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for minimal cleaning of lead scoring data.
    
    Removes duplicates and replaces 'Select' with NaN.
    """

    def fit(self, X, y=None):
        """
        Fit (no-op, stateless).
        
        Parameters:
            X: Input data (ignored).
            y: Targets (optional, ignored).
        
        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Apply cleaning: drop duplicates, replace 'Select' with NaN.
        
        Parameters:
            X (pd.DataFrame): Input data.
        
        Returns:
            pd.DataFrame: Cleaned data.
        """
        X = X.copy()  # Avoid modifying original
        X.drop_duplicates(inplace=True)  # Remove duplicates
        X.replace('Select', np.nan, inplace=True)  # Handle placeholders as NaN
        return X
