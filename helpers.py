import joblib
import os
import pandas as pd
import sklearn.metrics
from sklearn.pipeline import Pipeline
import sys

# Cache the train and test data in {repo}/__data__.
cachedir = os.path.join(sys.path[0], '__data__')
memory = joblib.Memory(cachedir=cachedir, verbose=0)

def execute_on_pipeline(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict_proba(X_test)
    # Check that the predicted probabilities have an sklearn-compatible shape.
    assert (y_pred.ndim == 1) or \
           (y_pred.ndim == 2 and y_pred.shape[1] == 2), \
        'The predicted probabilities should match sklearn''s ' \
        '`predict_proba` output shape.'
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
    # Evaluate the predictions with the AUC of the ROC curve.
    return sklearn.metrics.roc_auc_score(y_test, y_pred)

@memory.cache()
def get_data(subset='train'):
    df = get_dataset_as_df(subset)
    # Split into feature matrix X and labels y.
    df.earns_over_50K = df.earns_over_50K.str.contains('>').astype(int)
    X, y = df.drop(['earns_over_50K'], axis=1), df.earns_over_50K
    return X, y


def get_dataset_as_df(subset):
    # Construct the data URL.
    csv_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/'
    csv_url += f'adult/adult.{"data" if subset == "train" else "test"}'
    # Define the column names.
    names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'earns_over_50K']
    # Read the CSV.
    print(f'Downloading {subset} dataset to __data__/ ...')
    df = pd.read_csv(
        csv_url,
        sep=', ',
        names=names,
        skiprows=int(subset == 'test'),
        na_values='?')
    return df

def to_pipeline(step):
    return Pipeline([step])
