from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import category_encoders as ce

class LeadScoringPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.target_encoder = None
        self.binary_cols = [
            'Do Not Email', 'A free copy of Mastering The Interview',
            'Search', 'Newspaper Article', 'X Education Forums',
            'Newspaper', 'Digital Advertisement', 'Through Recommendations',"What is your current occupation"
        ]
        self.target_encoder_cols = ['Lead Source', 'Lead Origin','Last Activity','Specialization']
        self.drop_cols = [
            'Prospect ID', 'Lead Number', 'How did you hear about X Education', 'Lead Profile',
            'Lead Quality', 'Asymmetrique Profile Score', 'Asymmetrique Activity Score',
            'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Tags',
            'Last Notable Activity', 'City', 'Country', 'What matters most to you in choosing a course',
            'Magazine', 'Receive More Updates About Our Courses', 'Update me on Supply Chain Content',
            'Get updates on DM Content', 'I agree to pay the amount through cheque','Do Not Call'
        ]
        self.rare_last_activity = None

    def fit(self, X, y=None):
        X = X.copy()
        X.replace('Select', np.nan, inplace=True)

        # Map values
        X['Lead Source'] = self._map_lead_source(X['Lead Source'])
        X['Lead Origin'] = self._map_lead_origin(X['Lead Origin'])

        # Handle rare Last Activity
        last_activity_counts = X['Last Activity'].value_counts()
        self.rare_last_activity = last_activity_counts[last_activity_counts < 10].index.tolist()

        # Fit target encoder
        self.target_encoder = ce.TargetEncoder(cols=self.target_encoder_cols)
        self.target_encoder.fit(X[self.target_encoder_cols], y)

        return self

    def transform(self, X):
        X = X.copy()
        X.drop_duplicates(inplace=True)
        X.replace('Select', np.nan, inplace=True)

        # Map categorical values
        X['Lead Source'] = self._map_lead_source(X['Lead Source'])
        X['Lead Origin'] = self._map_lead_origin(X['Lead Origin'])

        # Fill nulls
        X['Specialization'] = X['Specialization'].fillna('Others')
        X['What is your current occupation'] = X['What is your current occupation'].fillna('Unknown')
        X['Last Activity'] = X['Last Activity'].fillna('Email Opened')
        X['TotalVisits'] = X['TotalVisits'].fillna(X['TotalVisits'].median())
        X['Page Views Per Visit'] = X['Page Views Per Visit'].fillna(X['Page Views Per Visit'].median())
        X['Total Time Spent on Website'] = X['Total Time Spent on Website'].fillna(X['Total Time Spent on Website'].median())

        # Group rare classes in Last Activity
        if self.rare_last_activity:
            X['Last Activity'] = X['Last Activity'].replace(self.rare_last_activity, 'Others')

        # Encode target-aware columns
        X[self.target_encoder_cols] = self.target_encoder.transform(X[self.target_encoder_cols])

        # One-hot encode binary columns
        # X = pd.get_dummies(X, columns=self.binary_cols, drop_first=True)

        # Drop uninformative or noisy columns
        X.drop(columns=self.drop_cols, errors='ignore', inplace=True)

        # Optionally drop Do Not Call if extremely imbalanced
        if 'Do Not Call' in X.columns and X['Do Not Call'].value_counts(normalize=True).get('Yes', 0) < 0.01:
            X.drop(columns=['Do Not Call'], inplace=True)

        return X

    def _map_lead_source(self, series):
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
        return series.replace({
            'Lead Import': 'Other',
            'Quick Add Form': 'Other'
        }).fillna('Other')
