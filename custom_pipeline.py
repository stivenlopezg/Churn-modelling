import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utilities.config import geography_categories, gender_categories, card_and_member_categories


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.loc[:, self.columns]
        return X


class ConvertDtypes(BaseEstimator, TransformerMixin):
    def __init__(self, numerical: list, categorical: list):
        if not isinstance(numerical, list):
            self.numerical = [numerical]
        else:
            self.numerical = numerical
        if not isinstance(categorical, list):
            self.categorical = [categorical]
        else:
            self.categorical = categorical

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for numerical in self.numerical:
            X[numerical] = pd.to_numeric(X[numerical])
        for categorical in self.categorical:
            if categorical == 'Geography':
                categories = geography_categories
            elif categorical == 'Gender':
                categories = gender_categories
            else:
                categories = card_and_member_categories
            X[categorical] = pd.Categorical(X[categorical], categories=categories)
        return X


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(X, columns=self.columns, drop_first=True)
        return X


class GetDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.columns)
        return X


class OutlierDummies(BaseEstimator, TransformerMixin):
    def __init__(self, distribution: str = 'gaussian', tail: str = 'right', fold: float = 3, variables: list = None):
        if distribution not in ['gaussian', 'skewed']:
            raise ValueError('distribution takes only values "gaussian", or "skewed"')
        if tail not in ['right', 'left', 'both']:
            raise ValueError('tail takes only values "right", "left", or "both"')
        if fold <= 0:
            raise ValueError('fold takes only positive numbers')
        self.tail = tail
        self.fold = fold
        self.variables = variables
        self.distribution = distribution
        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if self.tail in ['right', 'both']:
            if self.distribution == 'gaussian':
                self.right_tail_caps_ = (X[self.variables].mean() + self.fold * X[self.variables].std()).to_dict()
            else:
                iqr = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.right_tail_caps_ = (X[self.variables].quantile(0.75) + iqr * self.fold).to_dict()
        if self.tail in ['left', 'both']:
            if self.distribution == 'gaussian':
                self.left_tail_caps_ = (X[self.variables].mean() - self.fold * X[self.variables].std()).to_dict()
            else:
                iqr = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.left_tail_caps_ = (X[self.variables].quantile(0.25) - (iqr * self.fold)).to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        for variable in self.right_tail_caps_.keys():
            if self.distribution == 'gaussian':
                X[f'{variable}_gauss_rind'] = X[variable] \
                    .apply(lambda x: 1 if x > self.right_tail_caps_[variable] else 0)
            else:
                X[f'{variable}_skewed_rind'] = X[variable] \
                    .apply(lambda x: 1 if x > self.right_tail_caps_[variable] else 0)
        for variable in self.left_tail_caps_.keys():
            if self.distribution == 'gaussian':
                X[f'{variable}_gauss_lind'] = X[variable] \
                    .apply(lambda x: 1 if x < self.left_tail_caps_[variable] else 0)
            else:
                X[f'{variable}_skewed_lind'] = X[variable] \
                    .apply(lambda x: 1 if x < self.left_tail_caps_[variable] else 0)
        return X
