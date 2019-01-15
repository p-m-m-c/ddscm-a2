# OOP implementation of simulation scripts
import math
import json
import pickle
import pandas as pd
import numpy as np
from collections import deque
from load_model import load_trained_xgb_regressor

class BaseSimulation:
    """
    Base simulation class that serves as super class for the moving
    average and XGB-driven prediction simulations.
    """

    def __init__(self, path_to_input_data, **kwargs):
        """
        Path_to_input_data is the relative path to input data
        **kwargs can contain any of the following parameters (if not specified, they take default values):
         - shelf life m
         - cost of shortage u
         - cost of waste w
         - safety factor z
        """

        self.path_to_input_data = path_to_input_data

        # Setting the default parameter values for m, w, u and z
        self._m = kwargs.get('m', 5)
        self._u = kwargs.get('u', 500)
        self._w = kwargs.get('w', 100)
        self._z = kwargs.get('z', 2)

        # Setting 0 values for statistics to be printed
        self.shortage_pct = 0
        self.waste_pct = 0
        self.daily_costs = 0
        return

    def prepare_data(self):
        """
        Prepare the data into a pandas DataFrame for further
        processing. Steps:
         - read data
         - parse date column
         - rename columns for brevity
        """

        df = pd.read_excel(self.path_to_input_data)
        df['Date'] = pd.to_datetime(df.Date, format='%Y%m%d')
        df.rename(columns={'# items demanded': 'demand',
                           'Avg temp in 0.1oC': 'temperature',
                           'Rainfall in 24h in 0.1mm': 'precipitation',
                           }, inplace=True)
        self._data = df
        return

    def run(self):
        """
        Run the simulation on the data contained in self
        """

        # Assign demand vector (from csv) to D_obs: this is observed demand
        self._D_obs = self._data['demand'].values
        self._D_fcst = self._data['demand_pred'].values

        def S_rmse_alt(d, squared_errors, z):
            """
            Determine order up to level s based on demand for today and tomorrow
            d, the errors and the safety factor (z)
            """
            rmse = math.sqrt(np.mean(squared_errors))
            return round(d + z * rmse)

        I = deque(5 * [0])  # Initialise inventory deque of only 0s
        self._Twarm = 14  # Number of warmup days
        self._TotOrdered = 0  # Initialise total ordered
        self._TotDemand = 0  # Initialise total demand
        self._NShort = 0  # Initialise shortage
        self._TotWaste = 0 # Initialise total waste amount
        self._TC = 0  # Initialise total cost variable
        fillrateDay = []  # Initialise fill rate per day array
        squared_errors = []  # Initialise empty array for squared errors per day
        today_short = 0  # Initialise zero for today shortage

        for t in range(self._Twarm, len(self._D_obs) - 1):
            mu_t = self._D_fcst[t] # Extract demand pred for today
            mu_t1 = self._D_fcst[t + 1] # Extract demand pred for tomorrow
            mu_lr = mu_t + mu_t1  # Forecast demand for today and tomorrow summed

            #   Calculate squared error based on d_{t} and d_{t+1} and append to list
            #   of squared errors
            demand_t_and_t_plus_1_true = self._D_obs[t] + self._D_obs[t + 1]
            squared_error = (demand_t_and_t_plus_1_true - mu_lr)**2
            squared_errors.append(squared_error)

            # Determine s_t based on RMSE-like fcst error
            s_t = S_rmse_alt(mu_lr, squared_errors, self._z)
            q = max(0, (s_t - sum(I)))  # Determine order quantity q for next day
            d_observed = self._D_obs[t]  # Read value from array of demand D_obs
            self._TotDemand += d_observed  # Add current demand to total to obtain sum after
            # warm-up period

            # Determine fill rate for the day
            if d_observed == 0:
                fillrateDay.append(1)  # Set fill rate to 1 in case demand is 0

            elif d_observed >= sum(I):  # If demand is larger than total inventory
                fillrateDay.append(sum(I) / d_observed)  # Append fill rate to fill rate list
                today_short = d_observed - sum(I)  # We have shortage: lost sales
                self._NShort += today_short  # Add shortage to total shortage
                I = deque(5 * [0])  # Reset I to only zeros, because we have sold out

            else:
                fillrateDay.append(1)
                for i in range(self._m):  # Inventory is taken full FIFO (old -> new)

                    # Procedure for handling demand; waterfall like approach
                    if I[i] > d_observed:  # If we have more inventory of exp date i than demand
                        I[i] = I[i] - d_observed  # Remaining inventory is lessened
                        d_observed = 0  # D observed is fulfilled, so set to 0 and break out of loop
                        break
                    else:  # If d_obs >= I[i]; handle demand and continue looping
                        d_observed = d_observed - I[i]
                        I[i] = 0

            # Determine stats at day's end
            today_waste = I.popleft()  # Count expired products as waste
            self._TotWaste += today_waste # Add waste for day to total waste
            self._TotOrdered += q  # Add ordered quantity to total ordered quantity
            self._TC += (self._u * today_short) + (self._w * today_waste)
            today_short = 0  # Reset today_short to 0 again as it's not always assigned

            I.append(q) # Add newly ordered goods to inventory as newest products (on the right)
        return

    def print_results(self):
        """
        Print the results of the simulation run
        """

        try:
            result_stats = {
                "Waste": round(self._TotWaste / self._TotOrdered, 3),
                "Short": round(self._NShort / self._TotDemand, 3),
                "Avg cost (day)": round(self._TC / (len(self._D_obs) - self._Twarm)),
            }

            print(json.dumps(result_stats, indent=3))
        except AttributeError:
            print("First run simulation before printing results")

        return

class MovingAvgSim(BaseSimulation):
    """
    Implementation for the moving average (MA) based simulation. Inherit from 
    BaseSimulation class. Predictions based on t-7- and t-14 average.
    """

    def __init__(self, path_to_input_data, **kwargs):
        super().__init__(path_to_input_data, **kwargs)
    
    def predict_demand(self, t=(7,14)):
        """Function to predict demand based on moving average for
        every day in t days back. Defaults to 7 and 14 i.e. one and two-weeks back."""

        # Define additional columns in with shifted demand by all of provided t days
        col_names = ('demand_t-{}'.format(t[0]), 'demand_t-{}'.format(t[1]))
        self._data[col_names[0]] = self._data['demand'].shift(t[0])
        self._data[col_names[1]] = self._data['demand'].shift(t[1])

        # Compute and round the average of the historical demand based on the number of columns
        # used for computing historical demand
        self._data['demand_pred'] = round((self._data[col_names[0]] + self._data[col_names[1]]) / len(t))
        del self._data[col_names[0]], self._data[col_names[1]] # Drop the columns again
        return

class XGBSim(BaseSimulation):
    """Implementation for the ML based simulation. Inherit from BaseSimulation class. 
    Predictions based on XGB regression. Model is pre-trained."""
    
    def __init__(self, path_to_input_data, **kwargs):
        super().__init__(path_to_input_data, **kwargs)

    def enrich_data(self):
        """
        Enriches self._data by:
        - renaming columns to easier names
        - adding weekday dummy columns
        - adding lagged variables
        - removes data that can lead to leakage: demand itself
        - removes data that XGB can't work with: the date column
        
        Output: None, only self._data is modified
        """

        _weekday_series = self._data['Date'].dt.day_name()
        _weekday_dummies = pd.get_dummies(_weekday_series)
        self._data = pd.concat([self._data, _weekday_dummies], 
                               axis=1)
        
        # Add lagged variables
        self._data['t-7'] = self._data['demand'].shift(7).fillna(self._data['demand'].mean())
        self._data['t-14'] = self._data['demand'].shift(14).fillna(self._data['demand'].mean())

    def predict_demand(self):
        """Function to predict demand based on pre-trained ML model
        for every day contained in self._data. Resulting vector is added to self._data as
        the demand_pred Series and used in subsequent run of the simulation."""

        self._model = load_trained_xgb_regressor()
        self._data['demand_pred'] = self._model.predict(self._data.drop(['Date', 'demand'], axis=1))
        return

MA_sim = MovingAvgSim(
    path_to_input_data='../data/MergedData2017.xlsx')
MA_sim.prepare_data()
MA_sim.predict_demand()
MA_sim.run()
MA_sim.print_results()

ML_sim = XGBSim(
    path_to_input_data='../data/MergedData2017.xlsx')
ML_sim.prepare_data()
ML_sim.enrich_data()
ML_sim.predict_demand()
ML_sim.run()
ML_sim.print_results()