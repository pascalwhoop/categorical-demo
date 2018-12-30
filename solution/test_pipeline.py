import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from unittest import TestCase

from solution.pipeline import get_category_preprocessing, get_preprocessing


class TestPipeline(TestCase):
    def test_get_category_preprocessing(self):
        pipeline = get_category_preprocessing(True, True)
        assert type(pipeline.steps[0][1]) is OneHotEncoder

    def test_get_preprocessing(self):
        pp = get_preprocessing(False)
        pp_pl = Pipeline(pp)
        pp_pl.fit(test_df)


test_data = [[39, 'State-gov', 77516, 'Bachelors', 13, 'Never-married',
              'Adm-clerical', 'Not-in-family', 'White', 'Male', 2174, 0, 40,
              'United-States', '<=50K'],
             [50, 'Self-emp-not-inc', 83311, 'Bachelors', 13,
              'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White',
              'Male', 0, 0, 13, 'United-States', '<=50K'],
             [38, 'Private', 215646, 'HS-grad', 9, 'Divorced',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', 0, 0, 40,
              'United-States', '<=50K'],
             [53, 'Private', 234721, '11th', 7, 'Married-civ-spouse',
              'Handlers-cleaners', 'Husband', 'Black', 'Male', 0, 0, 40,
              'United-States', '<=50K'],
             [28, 'Private', 338409, 'Bachelors', 13, 'Married-civ-spouse',
              'Prof-specialty', 'Wife', 'Black', 'Female', 0, 0, 40, 'Cuba',
              '<=50K'],
             [37, 'Private', 284582, 'Masters', 14, 'Married-civ-spouse',
              'Exec-managerial', 'Wife', 'White', 'Female', 0, 0, 40,
              'United-States', '<=50K'],
             [49, 'Private', 160187, '9th', 5, 'Married-spouse-absent',
              'Other-service', 'Not-in-family', 'Black', 'Female', 0, 0, 16,
              'Jamaica', '<=50K'],
             [52, 'Self-emp-not-inc', 209642, 'HS-grad', 9,
              'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White',
              'Male', 0, 0, 45, 'United-States', '>50K'],
             [31, 'Private', 45781, 'Masters', 14, 'Never-married',
              'Prof-specialty', 'Not-in-family', 'White', 'Female', 14084, 0,
              50, 'United-States', '>50K'],
             [42, 'Private', 159449, 'Bachelors', 13, 'Married-civ-spouse',
              'Exec-managerial', 'Husband', 'White', 'Male', 5178, 0, 40,
              'United-States', '>50K']]

test_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'earns_over_50K']

test_df = pd.DataFrame(test_data, columns=test_columns)
