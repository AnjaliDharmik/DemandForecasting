import os
import sys
import numpy as np 
import pandas as pd
import dill

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_val_score

from src.exception import CustomException
from src.logger import logging

def model_metrics(true, predicted):
    try :
        MAE = mean_absolute_error(true, predicted)
        MSE = mean_squared_error(true, predicted)
        RMSE = np.sqrt(mean_squared_error(true, predicted))
          
        return MAE, MSE, RMSE
        
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)
        
def hyperparameter_tuning(X_train,y_train,model,param_grid,cv=5,n_iter=100):
    try:    
            # Create the RandomizedSearchCV object
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter,
                                       scoring='neg_mean_squared_error', cv=cv, random_state=42, n_jobs=-1,verbose=1)
            
            # Fit the model to find the best hyperparameters
            random_search.fit(X_train, y_train)
            
            # Get the best hyperparameters
            best_params = random_search.best_params_

            print("Best Hyperparameters:", best_params)
 
    except Exception as e:
        logging.info('Exception occured during hyperparameter tuning')
        raise CustomException(e,sys)
    
    return best_params
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        
        report = []
        
        # Loop over each model
        for model_name, model_info in models.items():
            
            logging.info('Hyperparameter tuning started')
           
            model = model_info['model']
            params = model_info['params']
            
             # Perform Hyperparameter Selection
            best_params = hyperparameter_tuning(X_train, y_train,model,params,cv=5,n_iter=30)
            
            try:
                # Train the model with the best hyperparameters
                model = model(**best_params)
            except:
                pass
            
            # Train the model on the entire training set
            model.fit(X_train, y_train)

            # Predict Training data
            y_train_pred = model.predict(X_train)

            # Make predictions on the test data
            y_pred = model.predict(X_test)
            
            MAE, MSE, RMSE = model_metrics(y_train,y_train_pred)
            print('\nTraining Mean Absolute Error:', MAE)
            print('Training Mean Squared Error:', MSE)
            print('Training Root Mean Squared Error:', RMSE)

            # Evaluate the model on the test set
            # Calculate the Mean Squared Error
            MAE, MSE, RMSE = model_metrics(y_test,y_pred)
            print('\nTesting Mean Absolute Error:', MAE)
            print('Testing Mean Squared Error:', MSE)
            print('Testing Root Mean Squared Error:', RMSE)

            report.append([model,MAE,MSE,RMSE])
        
        df_results = pd.DataFrame(report, columns=['Model Name', 'Mean Absolute Error',\
                        'Mean Squared Error','Root Mean Squared Error'])\
        .sort_values(by=["Root Mean Squared Error"])

        df_results.to_csv('model_report.csv',index=False,header=True)

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
   