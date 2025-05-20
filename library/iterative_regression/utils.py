import pandas as pd
from typing import Optional
import numpy as np
from config.config import config
import pickle

def ahi_class(x: pd.Series, num_class: Optional[int] = 4) -> pd.Series:
    """
    Apply threshold to the AHI.
    :param x: pd.Series, continuous column of the target, in this method, the ahi
    :param num_class: number of classes we want to transform the continuous input
    :return:
    """

    if num_class == 4:
        ahi_range = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}

        def threshold_ahi(ahi: float, ahi_range: dict):
            """Apply the threshold to the AHI column."""
            if pd.notna(ahi):
                if ahi < 5:
                    return ahi_range['Normal']
                if 5 <= ahi < 15:
                    return ahi_range['Mild']
                if 15 <= ahi < 30:
                    return ahi_range['Moderate']
                if ahi >= 30:
                    return ahi_range['Severe']
            return np.nan

        x = x.apply(func=lambda ahi: threshold_ahi(ahi, ahi_range)).astype(float)

    if num_class == 2:
        ahi_range = {'Normal_mild': 0, 'Moderate_severe': 1}

        def threshold_ahi(ahi: float, ahi_range: dict):
            """Apply the threshold to the AHI column."""
            if pd.notna(ahi):
                if 0 <= ahi < 15:
                    return ahi_range['Normal_mild']
                if ahi >= 15:
                    return ahi_range['Moderate_severe']
            return np.nan

        x = x.apply(func=lambda ahi: threshold_ahi(ahi, ahi_range)).astype(float)

    if num_class == 3:
        ahi_range = {'Normal': 0, 'Mild': 1, 'Moderate_Severe': 2, }

        def threshold_ahi(ahi: float, ahi_range: dict):
            """Apply the threshold to the AHI column."""
            if pd.notna(ahi):
                if ahi < 10:
                    return ahi_range['Normal']
                if 10 <= ahi < 20:
                    return ahi_range['Mild']
                if ahi >= 20:
                    return ahi_range['Moderate_severe']
            return np.nan

        x = x.apply(func=lambda ahi: threshold_ahi(ahi, ahi_range)).astype(float)


    return x

def log_transform_nans(x:float)-> float:
    """Log transform able to hande nan values ina frame column"""
    if not np.isnan(x):
        return np.log1p(x)
    else:
        return x


def get_ordinal_encoding_log() -> dict:
    """
    Get the dictionary on which responses labels were encoded into each ordinal response of the questionnaire
    :return:
        dict, nested dictionary.
            Outer Keys:
                '<column_name_questionnaire>'
            Inner Keys:
                'definition': which definition from the config was utilized for the ordinal encoding
                'encoding' the mapping from the strings to the numerical values
    """
    # Open the pickle file in binary mode for reading
    with open(config.get('ordinal_encoding_log'), 'rb') as f:
        # Load the data from the pickle file
        data = pickle.load(f)
    return data