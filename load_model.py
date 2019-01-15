import pickle

def load_trained_xgb_regressor():
    """
    Load the trained XGB regressor model from the pickle file. 
    Be sure to do this BEFORE creating an instance of the 
    DemandForecast class, because a trained model is a property
    of the DemandForecast class.
    
    Returns: Trained XGBRegressor model, used for predictions on new data
    """
    return pickle.load(open("./xgb_tuned.p", "rb"))
