# Spam Detector Project
This project involves building two classification models to detect spam: a Logistic Regression model and a Random Forest model. The goal is to evaluate which model performs better in identifying spam using the provided dataset. The project is structured into two main components: a Jupyter Notebook for exploratory data analysis and processing, and a Python script (`pipeline.py`) to handle the pipeline of tasks in a modular and efficient manner.

## Project Structure
- **`spam_detector.ipynb`**: This Jupyter Notebook contains the step-by-step process of loading the data, exploring it, preprocessing, and building and evaluating the models. It provides a detailed walkthrough of the analysis and includes visualizations to aid understanding.

- **`pipeline.py`**: This Python script encapsulates the entire workflow into a streamlined pipeline using `scikit-learn`'s `Pipeline` class. It includes functions for loading data, preprocessing, building models, and evaluating their performance.

## Workflow Overview
1. **Data Loading**: The dataset is loaded from a CSV file hosted online.
2. **Data Preprocessing**:
   - The dataset is split into features (`X`) and labels (`y`).
   - The data is further split into training and testing sets.
   - The features are scaled using `StandardScaler`.
3. **Model Building**:
   - Two models are built: a Logistic Regression model and a Random Forest Classifier.
   - Each model is evaluated using accuracy as the performance metric.
4. **Pipeline Construction**:
   - A pipeline is created for each model, combining the scaling and model-fitting steps.
   - The pipelines are used to train and test the models, ensuring a consistent workflow.

## Results
- **Logistic Regression Accuracy**: 0.9278
- **Random Forest Classifier Accuracy**: 0.9669

As hypothesized, the Random Forest Classifier outperformed the Logistic Regression model, likely due to its ability to handle complex, non-linear relationships within the data.

## Files
- **`spam_detector.ipynb`**: The Jupyter Notebook that details the entire process from data loading to model evaluation.
- **`pipeline.py`**: The Python script that implements the workflow using `scikit-learn`'s `Pipeline` class.
- **`README.md`**: This file.
- **`LICENSE`**: The Unlicence

## Dependencies
- **pandas**: Data manipulation and analysis.
- **scikit-learn**: Machine learning library for data preprocessing, model building, and evaluation.

## Installation
1. Clone the repository
2. Install the required packages
3. Run the Jupyter Notebook, `spam_detector.ipynb`
4. Run the pipeline script, `pipeline.py`

## Author
Files created by David Kaplan