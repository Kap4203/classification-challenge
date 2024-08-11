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
    return train_test_split(X, y, random_state=1)


def build_pipeline(model):
    """Build pipeline with scaling and model"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])


def evaluate_model(pipline, X_train, X_test, y_train, y_test):
    """Fit the pipeline, make prediction, and evelaute the model"""
    pipline.fit(X_train, y_train)
    prediction = pipline.predict(X_test)
    return accuracy_score(y_test, prediction)

def main():
    # Load the data
    url = "https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv"
    data = load_data(url)


    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)


    # Logistic Regression
    lr_pipeline = build_pipeline(LogisticRegression(random_state=1))
    lr_accuracy = evaluate_model(lr_pipeline, X_train, X_test, y_train, y_test)
    print(f"Logistic Regression accuracy is: {lr_accuracy}")


    # Random Forest Classifier
    rf_pipeline = build_pipeline(RandomForestClassifier(random_state=1))
    rf_accuracy = evaluate_model(rf_pipeline, X_train, X_test, y_train, y_test)
    print(f"Random Forest Classifier accuracy is: {rf_accuracy}")


if __name__ == "__main__":
    main()