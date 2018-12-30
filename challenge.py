import joblib
import os
import pandas as pd
import sklearn.metrics
import sklearn.pipeline
import sys

# Cache the train and test data in {repo}/__data__.
from sklearn.model_selection import GridSearchCV

from settings import settings

cachedir = os.path.join(sys.path[0], '__data__')
memory = joblib.Memory(cachedir=cachedir, verbose=0)


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


def score_solution():
    # Ask the solution for the model pipeline.
    import solution
    pipeline = solution.get_pipeline()
    error_message = 'Your `solution.get_pipeline` implementation should ' \
                    'return an `sklearn.pipeline.Pipeline`.'
    assert isinstance(pipeline, sklearn.pipeline.Pipeline), error_message
    # Train the model on the training DataFrame.
    X_train, y_train = get_data(subset='train')
    pipeline.fit(X_train, y_train)
    # Apply the model to the test DataFrame.
    X_test, y_test = get_data(subset='test')
    y_pred = pipeline.predict_proba(X_test)
    # Check that the predicted probabilities have an sklearn-compatible shape.
    assert (y_pred.ndim == 1) or \
           (y_pred.ndim == 2 and y_pred.shape[1] == 2), \
        'The predicted probabilities should match sklearn''s ' \
        '`predict_proba` output shape.'
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
    # Evaluate the predictions with the AUC of the ROC curve.
    joblib.dump(pipeline, 'pipeline.dump')
    return sklearn.metrics.roc_auc_score(y_test, y_pred)


def grid_search():
    from solution import get_pipeline

    s = settings()
    grid_params = s['GRID_PARAMS']
    pipeline = get_pipeline()
    search = GridSearchCV(pipeline, grid_params, n_jobs=8)
    X_train, y_train = get_data(subset='train')
    results = search.fit(X_train, y_train)
    print(search.best_params_)
    print("====='")
    print(search.cv_results_)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        grid_search()
    else:
        print(score_solution())
