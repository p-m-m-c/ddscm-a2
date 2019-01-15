## Machine learning in supply chain simulation

For this assignment, we had to compare the outcomes of two demand forecasting approaches:
- A historical (moving) average approach;
- An approach based on machine learning (in my case, XGBoost).

This repository contains the core of the assignment and contains the following files:
- EDA.ipynb: a notebook used for data inspection and creation of visualisations;
- xgb_trained.p: a binary file containing the trained XGBoost model;
- z_results_df.p: a binary file containing an optimization on the safety factor z;
- Simulation_class.py: implementation of the moving average and machine learning simulations.
