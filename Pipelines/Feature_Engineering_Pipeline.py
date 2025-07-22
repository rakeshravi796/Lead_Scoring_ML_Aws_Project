# import pandas as pd
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# import category_encoders as ce


# class LeadScoringFeatureEngineer(BaseEstimator, TransformerMixin):
#     """
#     Custom transformer for feature engineering in lead scoring pipelines.
    
#     Handles mapping, encoding, imputation, and dropping columns.
#     """

#     def __init__(self):
#         """
#         Initialize with encoders, column lists, and rare activity tracker.
#         """
#         self.target_encoder = None
#         self.ohe_cols = [
#             'Do Not Email', 'A free copy of Mastering The Interview',
#             'Search', 'Newspaper Article', 'X Education Forums',
#             'Newspaper', 'Digital Advertisement', 'Through Recommendations',
#             "What is your current occupation"
#         ]
#         self.target_encoder_cols = ['Lead Source', 'Lead Origin', 'Last Activity', 'Specialization']
#         self.drop_cols = [
#             'Prospect ID', 'Lead Number', 'How did you hear about X Education', 'Lead Profile',
#             'Lead Quality', 'Asymmetrique Profile Score', 'Asymmetrique Activity Score',
#             'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Tags',
#             'Last Notable Activity', 'City', 'Country', 'What matters most to you in choosing a course',
#             'Magazine', 'Receive More Updates About Our Courses', 'Update me on Supply Chain Content',
#             'Get updates on DM Content', 'I agree to pay the amount through cheque', 'Do Not Call'
#         ]
#         self.rare_last_activity = None

#     def fit(self, X, y=None):
#         """
#         Fit target encoder and identify rare categories.
        
#         Parameters:
#             X (pd.DataFrame): Input features.
#             y (array-like, optional): Target values.
        
#         Returns:
#             self
#         """
#         X = X.copy()
#         X.replace('Select', np.nan, inplace=True)  # Handle placeholders

#         # Map categorical values
#         X['Lead Source'] = self._map_lead_source(X['Lead Source'])
#         X['Lead Origin'] = self._map_lead_origin(X['Lead Origin'])

#         # Identify rare 'Last Activity' categories (<10 occurrences)
#         last_activity_counts = X['Last Activity'].value_counts()
#         self.rare_last_activity = last_activity_counts[last_activity_counts < 10].index.tolist()

#         # Impute missing values
#         X['Specialization'] = X['Specialization'].fillna('Others')
#         X['What is your current occupation'] = X['What is your current occupation'].fillna('Unknown')
#         X['Last Activity'] = X['Last Activity'].fillna('Email Opened')

#         # Group rare activities
#         if self.rare_last_activity:
#             X['Last Activity'] = X['Last Activity'].replace(self.rare_last_activity, 'Others')

#         # Fit target encoder
#         self.target_encoder = ce.TargetEncoder(cols=self.target_encoder_cols)
#         self.target_encoder.fit(X[self.target_encoder_cols], y)
#         return self

#     def transform(self, X):
#         """
#         Apply mappings, imputation, encoding, and drop columns.
        
#         Parameters:
#             X (pd.DataFrame): Input features.
        
#         Returns:
#             pd.DataFrame: Transformed features.
#         """
#         X = X.copy()
#         X.drop_duplicates(inplace=True)  # Remove duplicates
#         X.replace('Select', np.nan, inplace=True)  # Handle placeholders

#         # Map categorical values
#         X['Lead Source'] = self._map_lead_source(X['Lead Source'])
#         X['Lead Origin'] = self._map_lead_origin(X['Lead Origin'])

#         # Impute missing values
#         X['Specialization'] = X['Specialization'].fillna('Others')
#         X['What is your current occupation'] = X['What is your current occupation'].fillna('Unknown')
#         X['Last Activity'] = X['Last Activity'].fillna('Email Opened')
#         X['TotalVisits'] = X['TotalVisits'].fillna(X['TotalVisits'].median())
#         X['Page Views Per Visit'] = X['Page Views Per Visit'].fillna(X['Page Views Per Visit'].median())
#         X['Total Time Spent on Website'] = X['Total Time Spent on Website'].fillna(X['Total Time Spent on Website'].median())

#         # Group rare activities (using fitted values)
#         if self.rare_last_activity:
#             X['Last Activity'] = X['Last Activity'].replace(self.rare_last_activity, 'Others')

#         # Apply target encoding
#         X[self.target_encoder_cols] = self.target_encoder.transform(X[self.target_encoder_cols])

#         # Drop specified columns
#         X.drop(columns=self.drop_cols, errors='ignore', inplace=True)

#         # One-hot encode specified columns
#         for col in self.ohe_cols:
#             if col in X.columns:
#                 dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
#                 X = pd.concat([X.drop(columns=col), dummies], axis=1)

#         return X

#     def _map_lead_source(self, series):
#         """
#         Map lead sources to grouped categories, fill NaN with 'Other'.
        
#         Parameters:
#             series (pd.Series): Lead Source column.
        
#         Returns:
#             pd.Series: Mapped series.
#         """
#         mapping = {
#             'google': 'Google',
#             'bing': 'Search Engine',
#             'Google': 'Search Engine',
#             'Organic Search': 'Search Engine',
#             'Click2call': 'Other',
#             'Live Chat': 'Other',
#             'Social Media': 'Other',
#             'Press_Release': 'Other',
#             'Pay per Click Ads': 'Other',
#             'blog': 'Other',
#             'WeLearn': 'Other',
#             'welearnblog_Home': 'Other',
#             'youtubechannel': 'Other',
#             'testone': 'Other',
#             'NC_EDM': 'Other'
#         }
#         return series.replace(mapping).fillna('Other')

#     def _map_lead_origin(self, series):
#         """
#         Map lead origins to grouped categories, fill NaN with 'Other'.
        
#         Parameters:
#             series (pd.Series): Lead Origin column.
        
#         Returns:
#             pd.Series: Mapped series.
#         """
#         return series.replace({
#             'Lead Import': 'Other',
#             'Quick Add Form': 'Other'
#         }).fillna('Other')


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder  # NEW: Import for consistent OHE

class LeadScoringFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering in lead scoring pipelines.
    
    Handles mapping, encoding, imputation, and dropping columns.
    """
    
    def __init__(self):
        """
        Initialize with encoders, column lists, and rare activity tracker.
        """
        self.target_encoder = None
        self.onehot_encoder = None  # NEW: For consistent one-hot encoding
        self.ohe_feature_names = None  # NEW: To store expected OHE column names
        self.ohe_cols = [
            'Do Not Email', 'A free copy of Mastering The Interview',
            'Search', 'Newspaper Article', 'X Education Forums',
            'Newspaper', 'Digital Advertisement', 'Through Recommendations',
            "What is your current occupation"
        ]
        self.target_encoder_cols = ['Lead Source', 'Lead Origin', 'Last Activity', 'Specialization']
        self.drop_cols = [
            'Prospect ID', 'Lead Number', 'How did you hear about X Education', 'Lead Profile',
            'Lead Quality', 'Asymmetrique Profile Score', 'Asymmetrique Activity Score',
            'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Tags',
            'Last Notable Activity', 'City', 'Country', 'What matters most to you in choosing a course',
            'Magazine', 'Receive More Updates About Our Courses', 'Update me on Supply Chain Content',
            'Get updates on DM Content', 'I agree to pay the amount through cheque', 'Do Not Call'
        ]
        self.rare_last_activity = None
    
    def fit(self, X, y=None):
        """
        Fit target encoder, one-hot encoder, and identify rare categories.
        
        Parameters:
            X (pd.DataFrame): Input features.
            y (array-like, optional): Target values.
        
        Returns:
            self
        """
        X = X.copy()
        X.replace('Select', np.nan, inplace=True)  # Handle placeholders
        
        # Map categorical values
        X['Lead Source'] = self._map_lead_source(X['Lead Source'])
        X['Lead Origin'] = self._map_lead_origin(X['Lead Origin'])
        
        # Identify rare 'Last Activity' categories (<10 occurrences)
        last_activity_counts = X['Last Activity'].value_counts()
        self.rare_last_activity = last_activity_counts[last_activity_counts < 10].index.tolist()
        
        # Impute missing values
        X['Specialization'] = X['Specialization'].fillna('Others')
        X['What is your current occupation'] = X['What is your current occupation'].fillna('Unknown')
        X['Last Activity'] = X['Last Activity'].fillna('Email Opened')
        
        # Group rare activities
        if self.rare_last_activity:
            X['Last Activity'] = X['Last Activity'].replace(self.rare_last_activity, 'Others')
        
        # Fit target encoder
        self.target_encoder = ce.TargetEncoder(cols=self.target_encoder_cols)
        self.target_encoder.fit(X[self.target_encoder_cols], y)
        
        # NEW: Fit OneHotEncoder on OHE columns (learns all categories from training data)
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        self.onehot_encoder.fit(X[self.ohe_cols])
        
        # NEW: Store the expected feature names after OHE (for consistency in transform)
        self.ohe_feature_names = self.onehot_encoder.get_feature_names_out(self.ohe_cols)
        
        return self
    
    def transform(self, X):
        """
        Apply mappings, imputation, encoding, and drop columns.
        
        Parameters:
            X (pd.DataFrame): Input features.
        
        Returns:
            pd.DataFrame: Transformed features.
        """
        X = X.copy()
        X.drop_duplicates(inplace=True)  # Remove duplicates
        X.replace('Select', np.nan, inplace=True)  # Handle placeholders
        
        # Map categorical values
        X['Lead Source'] = self._map_lead_source(X['Lead Source'])
        X['Lead Origin'] = self._map_lead_origin(X['Lead Origin'])
        
        # Impute missing values
        X['Specialization'] = X['Specialization'].fillna('Others')
        X['What is your current occupation'] = X['What is your current occupation'].fillna('Unknown')
        X['Last Activity'] = X['Last Activity'].fillna('Email Opened')
        X['TotalVisits'] = X['TotalVisits'].fillna(X['TotalVisits'].median())
        X['Page Views Per Visit'] = X['Page Views Per Visit'].fillna(X['Page Views Per Visit'].median())
        X['Total Time Spent on Website'] = X['Total Time Spent on Website'].fillna(X['Total Time Spent on Website'].median())
        
        # Group rare activities (using fitted values)
        if self.rare_last_activity:
            X['Last Activity'] = X['Last Activity'].replace(self.rare_last_activity, 'Others')
        
        # Apply target encoding
        X[self.target_encoder_cols] = self.target_encoder.transform(X[self.target_encoder_cols])
        
        # Drop specified columns
        X.drop(columns=self.drop_cols, errors='ignore', inplace=True)
        
        # NEW: Apply fitted OneHotEncoder (ensures consistent columns)
        ohe_transformed = self.onehot_encoder.transform(X[self.ohe_cols])
        ohe_df = pd.DataFrame(ohe_transformed, columns=self.ohe_feature_names, index=X.index)
        
        # Concatenate OHE columns and drop originals
        X = pd.concat([X.drop(columns=self.ohe_cols), ohe_df], axis=1)
        
        # NEW: Reindex to ensure all expected OHE columns are present (fill missing with 0)
        expected_columns = [col for col in X.columns if col not in self.ohe_feature_names] + list(self.ohe_feature_names)
        X = X.reindex(columns=expected_columns, fill_value=0)
        
        return X
    
    def _map_lead_source(self, series):
        """
        Map lead sources to grouped categories, fill NaN with 'Other'.
        
        Parameters:
            series (pd.Series): Lead Source column.
        
        Returns:
            pd.Series: Mapped series.
        """
        mapping = {
            'google': 'Google',
            'bing': 'Search Engine',
            'Google': 'Search Engine',
            'Organic Search': 'Search Engine',
            'Click2call': 'Other',
            'Live Chat': 'Other',
            'Social Media': 'Other',
            'Press_Release': 'Other',
            'Pay per Click Ads': 'Other',
            'blog': 'Other',
            'WeLearn': 'Other',
            'welearnblog_Home': 'Other',
            'youtubechannel': 'Other',
            'testone': 'Other',
            'NC_EDM': 'Other'
        }
        return series.replace(mapping).fillna('Other')
    
    def _map_lead_origin(self, series):
        """
        Map lead origins to grouped categories, fill NaN with 'Other'.
        
        Parameters:
            series (pd.Series): Lead Origin column.
        
        Returns:
            pd.Series: Mapped series.
        """
        return series.replace({
            'Lead Import': 'Other',
            'Quick Add Form': 'Other'
        }).fillna('Other')
