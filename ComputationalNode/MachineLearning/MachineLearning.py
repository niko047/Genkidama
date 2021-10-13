#External Imports
import pandas as pd
import numpy as np

#Internal Imports
from .algorithm_switcher import algorithm_switch, ALGORITHM_SWITCHER
from .search_switcher import search_switch, SEARCH_SWITCHER

class BaseModel:

    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.DataFrame):
        self.X = X
        self.y = y
        BaseModel.data_validity_checks(X = X, y = y)

    @staticmethod
    def data_validity_checks(X, y) -> None:
        #Assert they're either pandas dataframes or pandas series
        assert isinstance(X, (pd.DataFrame, pd.Series, np.ndarray, np.array)) \
               and isinstance(y, (pd.DataFrame, pd.Series, np.ndarray, np.array))

        #Data must have same shape
        assert len(X) == len(y)

        #Data must not be empty
        assert (len(X) and len(y))

class HyperparameterTuning(BaseModel):

    def __init__(self,
                 X : pd.DataFrame,
                 y : pd.Series,
                 algorithm_str : str,
                 search_str : str,
                 hyperparams_grid : dict,
                 algorithm_kwargs = {}):

        super().__init__(X = X, y = y)
        self.algorithm_kwargs = algorithm_kwargs
        self.hyperparams_grid = hyperparams_grid
        self.algorithm_str = algorithm_str
        self.search_str = search_str

        self.validate_search()
        self.validate_algorithm()

    def validate_search(self) -> None:
        try:
            self.search = search_switch(self.search_str)
            assert self.search is not None
        except:
            raise Exception(f'Algorithm not available. Valid list is {SEARCH_SWITCHER}')

    def validate_algorithm(self) -> None:
        try:
            self.algorithm = algorithm_switch(self.algorithm_str)
            assert self.algorithm is not None
        except:
            raise Exception(f'Algorithm not available. Valid list is {ALGORITHM_SWITCHER}')

    def run_search(self) -> None:
        #TODO - Later on add **search_kwargs to customize it even more
        #Ideally we would like to be similar in parameters, so that we can pipeline it authomatically
        #But for now use if statements

        if self.search_str == 'grid_search_cv':
            return self.run_gridsearch_cv()

    def run_gridsearch_cv(self) -> dict:
        search_model = self.search(estimator = self.algorithm(**self.algorithm_kwargs),
                    param_grid=self.hyperparams_grid)
        fitted_search_model = search_model.fit(X = self.X, y = self.y)
        return {
            'best_params': fitted_search_model.best_params_,
            'best_score': fitted_search_model.best_score_,
        }




