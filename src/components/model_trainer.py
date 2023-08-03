#!/usr/bin/env python
# coding: utf-8

# Import Required Libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import *
from dataclasses import dataclass

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# Set the figure size for better visualization
plt.figure(figsize=(15, 8))

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Modelling
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge,Lasso
import xgboost as xgb
import lightgbm as lgb


# Initialize Model Trainer Configuration

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    model_report_file_path = os.path.join('artifacts','model_report.csv')
    


# # Step 3: Model Development

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,data):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            
            # Define Variables
            target_column_name = 'total_rooms_sold_lag_3'
            
            # Train Test Split
            X = data.drop([target_column_name],axis=1)
            y= data[target_column_name]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
         
            # Define the models and their hyperparameters
            model_report = []

            # Create the multiple linear regression model
            linear_reg_model = LinearRegression()

            # Perform cross-validation on the training set
            cv_scores = cross_val_score(linear_reg_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

            # Since cross_val_score returns negative mean squared error, take the absolute value
            mse_scores = -cv_scores
            rmse_scores = np.sqrt(mse_scores)

            # Train the multiple linear regression model on the entire training set
            linear_reg_model.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = linear_reg_model.predict(X_test)

            # Evaluate the model on the test set
            # Calculate the Mean Squared Error
            MAE = mean_absolute_error(y_test, y_pred)
            MSE = mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

            model_report.append(["LinearRegression",MAE,MSE,RMSE])
            
            # Define the hyperparameter grid for RandomizedSearchCV
            param_grid = {
                'alpha': np.logspace(-4, 2, 100),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }

            # Create the Ridge regression model
            ridge_model = Ridge()

            # Create the RandomizedSearchCV object
            random_search = RandomizedSearchCV(estimator=ridge_model, param_distributions=param_grid, n_iter=100,
                                               scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1)

            # Fit the model to find the best hyperparameters
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters found by RandomizedSearchCV
            best_alpha = random_search.best_params_['alpha']
            best_solver = random_search.best_params_['solver']


            # Train the Ridge regression model with the best hyperparameters
            ridge_model_best = Ridge(alpha=best_alpha, solver=best_solver)
            ridge_model_best.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = ridge_model_best.predict(X_test)

            # Calculate the Mean Squared Error
            MAE = mean_absolute_error(y_test, y_pred)
            MSE = mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
          

            model_report.append(["RidgeRegression",MAE,MSE,RMSE])
            
            # Create the Lasso regression model
            lasso_model = Lasso()

            # Define the hyperparameter grid for tuning
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]  # Values for the regularization strength
            }

            # Perform grid search with 5-fold cross-validation to find the best hyperparameters
            grid_search = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_alpha = grid_search.best_params_['alpha']

            # Create the Lasso regression model with the best hyperparameters
            best_lasso_model = Lasso(alpha=best_alpha)

            # Train the Lasso regression model on the entire training set
            best_lasso_model.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = best_lasso_model.predict(X_test)

            # Evaluate the model on the test set
            # Calculate the Mean Squared Error
            MAE = mean_absolute_error(y_test, y_pred)
            MSE = mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

            model_report.append(["LassoRegression",MAE,MSE,RMSE])
            
            # Define the hyperparameter grid for random search
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 1, 5]
            }

            # Define the XGBoost model
            xgb_model = xgb.XGBRegressor()

            # Perform Randomized Search Cross-Validation
            random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=10,\
                                               scoring='neg_mean_squared_error', cv=3, verbose=1, random_state=42)
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_params = random_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Train the model with the best hyperparameters
            xgb_model_best = xgb.XGBRegressor(**best_params)
            xgb_model_best.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = xgb_model_best.predict(X_test)

            # Evaluate the model
            # Calculate the Mean Squared Error
            MAE = mean_absolute_error(y_test, y_pred)
            MSE = mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
           
            model_report.append(["XGBRegressor",MAE,MSE,RMSE])
            
            # Define the LightGBM model
            lgb_model = lgb.LGBMRegressor()

            # Define the hyperparameter grid for random search
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }

            # Perform Randomized Search Cross-Validation
            random_search = RandomizedSearchCV(lgb_model, param_distributions=param_grid, n_iter=10,\
                                               scoring='neg_mean_squared_error', cv=3, verbose=1, random_state=42)
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_params = random_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Train the model with the best hyperparameters
            lgb_model_best = lgb.LGBMRegressor(**best_params)
            lgb_model_best.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = lgb_model_best.predict(X_test)

            # Evaluate the model
            # Calculate the Mean Squared Error
            MAE = mean_absolute_error(y_test, y_pred)
            MSE = mean_squared_error(y_test, y_pred)
            RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

            model_report.append(["LGBMRegressor",MAE,MSE,RMSE])

            
            model_report = evaluate_models(X_train,y_train,X_test,y_test,models)
            
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            # To get best model score 
            best_model_scores = min(model_report, key = lambda x: int(x[3]))

            best_model = best_model_scores[0]
            best_model_score = best_model_scores[-1] 
            
            print(f'Best Model Found , Model Name : {best_model} , RMSE : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model} , RMSE : {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info('Model pickle file saved')
            
            
            # Model Evaluation
            ytest_pred = best_model.predict(X_test)

            MAE, MSE, RMSE  = model_metrics(y_test, ytest_pred)
            logging.info(f'Test Mean Absolute Error : {MAE}')
            logging.info('Training Mean Squared Error:', MSE)
            logging.info(f'Test Root Mean Squared Error : {RMSE}')
            
            sns.distplot(y_test - ytest_pred)
         
            logging.info('Final Model Training Completed')
            
            return MAE, MSE, RMSE
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

