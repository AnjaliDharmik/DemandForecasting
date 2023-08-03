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

import warnings
warnings.filterwarnings('ignore')


# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    digital_visits_path: str = os.path.join('artifacts','digital_visits.csv')
    BK_LT_0_path: str = os.path.join('artifacts','hotel_bookings_at_leadtime_0.csv')
    BK_LT_3_path: str = os.path.join('artifacts','hotel_bookings_at_leadtime_3.csv')
    BK_LT_3_reserv_ch_path: str = os.path.join('artifacts','hotel_bookings_at_leadtime_3_by_reservation_channel.csv')
    Canc_LT_3_path: str = os.path.join('artifacts','recent_cancellations_at_leadtime_3.csv')
    hotel_details_path: str = os.path.join('artifacts','hotel_details.csv')


# Step 1: DataIngestion
# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Data ingestion method Started')
        try:
            
            # Replace 'YYYY-MM-DD' with your actual date format in the 'format' parameter
            date_parser = lambda x: pd.to_datetime(x, format='%d/%m/%Y')

            digital_visits_df = pd.read_csv('data/digital_visits.csv', parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_0_df = pd.read_csv('data/hotel_bookings_at_leadtime_0.csv', parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_3_df = pd.read_csv('data/hotel_bookings_at_leadtime_3.csv', parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_3_reserv_ch_df = pd.read_csv('data/hotel_bookings_at_leadtime_3_by_reservation_channel.csv', parse_dates=['stay_date'], date_parser=date_parser)
            Canc_LT_3_df = pd.read_csv('data/recent_cancellations_at_leadtime_3.csv', parse_dates=['stay_date'], date_parser=date_parser)
            hotel_details_df = pd.read_csv('data/hotel_details.csv')
         
            logging.info('Dataset read as pandas Dataframe')

            
            os.makedirs(os.path.dirname(self.ingestion_config.digital_visits_path),exist_ok=True)
            digital_visits_df.to_csv(self.ingestion_config.digital_visits_path,index=False)
            
            os.makedirs(os.path.dirname(self.ingestion_config.BK_LT_0_path),exist_ok=True)
            BK_LT_0_df.to_csv(self.ingestion_config.BK_LT_0_path,index=False)
            
            os.makedirs(os.path.dirname(self.ingestion_config.BK_LT_3_path),exist_ok=True)
            BK_LT_3_df.to_csv(self.ingestion_config.BK_LT_3_path,index=False)
            
            os.makedirs(os.path.dirname(self.ingestion_config.BK_LT_3_reserv_ch_path),exist_ok=True)
            BK_LT_3_reserv_ch_df.to_csv(self.ingestion_config.BK_LT_3_reserv_ch_path,index=False)
            
            os.makedirs(os.path.dirname(self.ingestion_config.Canc_LT_3_path),exist_ok=True)
            Canc_LT_3_df.to_csv(self.ingestion_config.Canc_LT_3_path,index=False)
            
            os.makedirs(os.path.dirname(self.ingestion_config.hotel_details_path),exist_ok=True)
            hotel_details_df.to_csv(self.ingestion_config.hotel_details_path,index=False)
            
            logging.info('Dataset saved in artifacts')
            
            return(
                self.ingestion_config.digital_visits_path,
                self.ingestion_config.BK_LT_0_path,
                self.ingestion_config.BK_LT_3_path,
                self.ingestion_config.BK_LT_3_reserv_ch_path,
                self.ingestion_config.Canc_LT_3_path,
                self.ingestion_config.hotel_details_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    

# Run Data ingestion
if __name__ == '__main__':
    obj = DataIngestion()
    digital_visits_df, BK_LT_0_df,BK_LT_3_df,BK_LT_3_reserv_ch_df,Canc_LT_3_df,hotel_details_df= obj.initate_data_ingestion()

    
    data_transformation = DataTransformation()
    data, _ = data_transformation.initate_data_transformation(digital_visits_df,                    BK_LT_0_df,BK_LT_3_df,BK_LT_3_reserv_ch_df,Canc_LT_3_df,hotel_details_df)

    modeltrainer = ModelTrainer()
    modeltrainer.initate_model_training( data)

