import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing, ensemble
from sklearn import tree
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from typing import List, Tuple

from settings import settings
from solution import categories
from solution.categories import get_categories


def get_pipeline(category_processing, input_dim, settings_=None):
    """
    This function should build an sklearn.pipeline.Pipeline object to train
    and evaluate a model on a pandas DataFrame. The pipeline should end with a
    custom Estimator that wraps a TensorFlow model. See the README for details.
    :param settings_:
    :param input_dim:
    :param category_processing: a callback function that has to return a Pipeline to process any categorical data
    """
    s = settings()
    # overwrite any env settings with explicit call parameters. probably not the best approach but hey. PoC right?
    if settings_:
        s = {**s, **settings_}

    pp = get_preprocessing(category_processing,  s['NORMALIZE'])
    classifier = get_classifier(input_dim=input_dim)
    steps = pp + classifier
    pipeline = Pipeline(steps)
    return pipeline


def get_classifier(**kwargs) -> List[Tuple[str, KerasClassifier]]:
    return [('keras', KerasClassifier(build_keras_classifier, epochs=3, verbose=2, **kwargs))]


def build_keras_classifier(input_dim=112, hidden_layers=3, hidden_units=24, dropout_rate=0.1, optimizer='adam'):
    model = Sequential()
    model.add(Dense(24, input_dim=input_dim))
    model.add(Dropout(rate=dropout_rate))
    for i in range(hidden_layers):
        model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mae', 'acc'])
    return model


#def get_category_preprocessing(onehotencode, normalize) -> Pipeline:
#    """
#    Handles all categories (string-based) in the dataset and converts them
#    :param onehotencode:
#    :return:
#    """
#    steps = []
#    if onehotencode:
#        steps.append(("ohe", preprocessing.OneHotEncoder(handle_unknown='ignore', categories=get_categories())))
#    else:
#        steps.append(("ordinal_enc", preprocessing.OrdinalEncoder(categories=get_categories())))
#    return Pipeline(steps)


def get_continuous_preprocessing(normalize):
    """
    Handles all continuous columns of the dataset
    :return:
    """
    steps = []
    if normalize:
        steps.append(("scale", StandardScaler()))
    else:
        steps.append(("nothing", FunctionTransformer(do_nothing, validate=False)))
    return Pipeline(steps)


def get_preprocessing(category_processing, normalize=True) -> List[Tuple]:
    """
    returns a list of preprocessing steps. can be nested
    :param onehotencode:
    :return:
    """
    steps = []

    # replace all NaN with "unknown" to mark them explicitly
    steps.append(("replace_nan", FunctionTransformer(mask_nan, validate=False)))
    steps.append(("preprocessing",
                  make_column_transformer(
                      (
                          category_processing,
                          ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                           'native-country'],
                      ),
                      (
                          get_continuous_preprocessing(normalize),
                          ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
                      )
                  )))

    return steps


def get_df_back(data, df_mask):
    """
    A small helper that gives me a df object again. odly enough, I can't use `return_df=True` on the `OrdinalEncoder`...
    :param data:
    :param df_mask:
    :return:
    """
    return pd.DataFrame(data, index=df_mask.index, columns=df_mask.columns)


def mask_nan(X, y=None):
    return X.fillna("unknown")


def do_nothing(X, y=None):
    return X
