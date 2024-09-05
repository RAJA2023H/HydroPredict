import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True
