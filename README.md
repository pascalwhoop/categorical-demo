# Machine Learning challenge

## Introduction

The goal of this challenge is to build a Machine Learning model to predict if a given adult's yearly income is above or below $50k.

To succeed, you must develop a `solution` Python package that implements a `get_pipeline` function that returns:

- [x] an [sklearn.pipeline.Pipeline](http://scikit-learn.org/stable/modules/pipeline.html)
- [x] that chains a series of [sklearn Transformers](http://scikit-learn.org/stable/data_transforms.html) to preprocess the data,
- [x] and ends with a [custom sklearn Estimator](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator) that wraps a [TensorFlow Estimator](https://www.tensorflow.org/get_started/custom_estimators),
- [x] and will be fed a pandas DataFrame of the [Adult Data Set](http://mlr.cs.umass.edu/ml/datasets/Adult) to train and evaluate the pipeline.

Note: to make this work, all your Transformers and final Estimator should operate on pandas DataFrames instead of numpy arrays. See sklearn's [rolling your own Estimator](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator) for more information on this topic.

## Getting started

1. Clone this repository (do not fork it!) and upload it to a fresh repository that you create.
2. Install [Miniconda](https://conda.io/miniconda.html) if you don't have it already.
3. Run `conda env create` from the repo's base directory to create the repo's conda environment from `environment.yml`. You may add packages listed on [anaconda.org](https://anaconda.org/) to `environment.yml` as desired.
4. Run `activate machine-learning-challenge-env` to activate the conda environment.
5. Start implementing the `def get_pipeline():` function in the `solution` directory!

## Evaluating your solution

To check your solution, run `python challenge.py` from the base of this repository. This will trigger the following steps:

1. Call `fitted_pipeline = solution.get_pipeline().fit(X_train, y_train)` where `X_train` is a pandas DataFrame and `y_train` is a pandas Series of labels.
2. Call `y_pred = fitted_pipeline.predict_proba(X_test)` where `X_test` is a pandas DataFrame of the same format as `X_train`.
3. Compute the ROC AUC between `y_pred` and `y_test` and print your score!

When you're ready, send us the URL to your repo!

## Stretch goals

If you really want to make an impression, try your hand at these stretch goals:

- [x] Use all of the provided features in your model.
- [x] Implement your pipeline so that you can [`joblib.dump`](https://pythonhosted.org/joblib/generated/joblib.dump.html) it to a file.
- [x] Find a non-trivial way of dealing with the missing values in the feature matrix.

Good luck!

-- radix.ai
