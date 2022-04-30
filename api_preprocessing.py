from doctest import DocFileCase
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from sklearn.preprocessing import StandardScaler

class ApiPreprocessing():
  def __init__(self):
    self.scaler = pickle.load(open('deploy/scaler.pkl', 'rb'))

  def data_preprocessing(self, df, response_json):
    # calculate age in days
    particular_date = datetime(response_json['year'], response_json['month'], response_json['day'])
    new_date_in_days = datetime.today() - particular_date

    # convert date columns to age in days column
    df['age'] = new_date_in_days
    df = df.drop(['year', 'month', 'day'], axis=1)

    # convert all types to int
    df = df.astype('int64')

    sc = self.scaler
    df_preprocessed = sc.fit_transform(df.values)

    return df_preprocessed

    return df