from datetime import datetime
import pandas as pd
import numpy as np
from .db import required_record_mapping

def read_csv(csv_file):
    """
    read csv and check input dataframe format
    Args:
        csv_file(string): path to csv_file
    Returns:
        df(dataframe): dataframe read from csv
    Raises:
        ValueError
    """
    # dtype = {n:t for n, t in required_record_mapping.items() if t == int or t == float}
    # converters = {n:pd.to_datetime for n, t in record_mapping.items() if t == datetime}
    try:
        df = pd.read_csv(csv_file)
    except Exception as error:
        raise ValueError('Error on reading csv: {}'.format(error))
    # check if all required columns in dataframe
    if not set(required_record_mapping.keys()).issubset(df.columns):
        raise ValueError('Missing required columns: {}'.format(set(required_record_mapping.keys()) - set(df.columns)))
    # change nan to None
    df = df.where((pd.notnull(df)), None)
    return df

