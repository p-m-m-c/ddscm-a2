# Import pandas for data manipulation
import pandas as pd
from math import sqrt

class DemandForecast:
    """
    Class used to import, transform and predict data. Provides friendly
    interface for usage in subsequent steps.
    """
    def __init__(self, file_path, prediction_model):
        self.file_path = file_path
        self.model = prediction_model
    
    def _create_data(self):
        """
        Read in the Excel file provided as self.file_path,
        return the raw data.
        
        Output: pd.DataFrame with data
        """
        _raw_data = pd.read_excel(self.file_path, 
                              parse_dates=['Date'])
        return _raw_data     
        
    def prepare_data(self):
        """
        Calls _create_data() and enriches self._data by:
        - renaming columns to easier names
        - adding weekday dumy columns
        - adding lagged variables
        - removes data that can lead to leakage: demand itself
        - removes data that XGB can't work with: the date column
        
        Output: None, only self._data is modified
        """
        self._data = self._create_data()
        # Add logging
        self._data.rename(columns={'# items demanded':'demand',
                   'Avg temp in 0.1oC': 'temperature',
                   'Rainfall in 24h in 0.1mm':'precipitation',
                  }, inplace=True)
        
        # Add logging // add weekday dummies
        _weekday_series = self._data['Date'].dt.day_name()
        _weekday_dummies = pd.get_dummies(_weekday_series)
        self._data = pd.concat([self._data, _weekday_dummies], 
                               axis=1)
        
        # Add logging // add lagged variables
        self._data['t-7'] = self._data['demand'].shift(7).fillna(self._data['demand'].mean())
        self._data['t-14'] = self._data['demand'].shift(14).fillna(self._data['demand'].mean())
        
    def predict_demand(self):
        """
        Save a predicted demand list for the entire self._data 
        DataFrame, based on the `predict` method of the trained XGB 
        regressor model. Only predict on valid columns (i.e. drop Date and demand 
        column to prevent data leakage or incompatibility of data type for XGB).
        
        Output: self._data['demand_pred'] pd Series
        """
        self._data['demand_pred'] = self.model.predict(self._data.drop(['Date',
                                        'demand'], axis=1))
        print("Predictions made")

    # Function to obtain demand prediction for day n and day n+1
    def get_prediction_for_day(self, n):
        """
        Obtain the predictions for day n (zero-based) and
        day n+1 as a tuple of floats.
        
        Input: integer n, indicating the day
        Output: (pred_n <float>, pred_n_plus_1 <float>)
        """
        assert n>=0 and n<len(self._data), "Provide 0 <= n < n_observations"
        
        _pred_day_n = self._data.loc[n, 'demand_pred']
        _pred_day_n_1 = self._data.loc[n+1, 'demand_pred']
        
        return _pred_day_n, _pred_day_n_1

    # Function to obtain true demand for day n and day n+1
    def get_observed_demand_for_day(self, n):
        """
        Obtain the true demand for day n (zero-based) and
        day n+1 as a tuple of floats.
        
        Input: integer n, indicating the day
        Output: (pred_n <float>, pred_n_plus_1 <float>)
        """
        assert n>=0 and n<len(self._data), "Provide 0 <= n < n_observations"
        
        _true_day_n = self._data.loc[n, 'demand']
        _true_day_n_1 = self._data.loc[n+1, 'demand']
        
        return _true_day_n, _true_day_n_1 