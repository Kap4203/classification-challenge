# Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def load_data(url):
    """Load data from a .csv file"""
    return pd.read_csv(url)


def preprocess_data(data):
    """Preprocess data by splitting into features and lables"""
    X = data.drop(columns='spam')
    y = data['spam']
    return train_test_split(X, y, random_state=11)


def build_pipeline(model):
    """Build pipeline with scaling and model"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])