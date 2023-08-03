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

from sklearn.preprocessing import LabelEncoder

# Initialize Data Transformation Configuration

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    merged_data_path: str = os.path.join('artifacts','data.csv')
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')


# Step 2: Data Transformation

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initate_data_transformation(self,digital_visits_path, BK_LT_0_path,BK_LT_3_path,BK_LT_3_reserv_ch_path,                                    Canc_LT_3_path,hotel_details_path):
    
        try:
            
            # Replace 'YYYY-MM-DD' with your actual date format in the 'format' parameter
            date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

            digital_visits_df = pd.read_csv(digital_visits_path, parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_0_df = pd.read_csv(BK_LT_0_path, parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_3_df = pd.read_csv(BK_LT_3_path, parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_3_reserv_ch_df = pd.read_csv(BK_LT_3_reserv_ch_path, parse_dates=['stay_date'], date_parser=date_parser)
            Canc_LT_3_df = pd.read_csv(Canc_LT_3_path, parse_dates=['stay_date'], date_parser=date_parser)
            hotel_details_df = pd.read_csv(hotel_details_path)
            
            # sample datasets
            logging.info("digital_visits:", digital_visits_df.head())
            logging.info("hotel_details:", hotel_details_df.head())
            logging.info("hotel_bookings_at_leadtime_3_by_reservation_channel:", BK_LT_3_reserv_ch_df.head())
            logging.info("recent_cancellations_at_leadtime_3:", Canc_LT_3_df.head())
            logging.info("bookings_leadtime_0_df:", BK_LT_0_df.head())
            logging.info("bookings_leadtime_3_df:", BK_LT_3_df.head())
            
            # size of relevant datasets
            logging.info("digital_visits shape:", digital_visits_df.shape)
            logging.info("hotel_details shape:", hotel_details_df.shape)
            logging.info("hotel_bookings_at_leadtime_3_by_reservation_channel shape:", BK_LT_3_reserv_ch_df.shape)
            logging.info("recent_cancellations_at_leadtime_3 shape:", Canc_LT_3_df.shape)
            logging.info("bookings_leadtime_0_df shape:", BK_LT_0_df.shape)
            logging.info("bookings_leadtime_3_df shape:", BK_LT_3_df.shape)
            
            # Check for duplicate data
            logging.info("Duplicate Data Summary:")
            logging.info("digital_visits :", digital_visits_df.duplicated().sum())
            logging.info("hotel_details :", hotel_details_df.duplicated().sum())
            logging.info("hotel_bookings_at_leadtime_3_by_reservation_channel :", BK_LT_3_reserv_ch_df.duplicated().sum())
            logging.info("recent_cancellations_at_leadtime_3 :", Canc_LT_3_df.duplicated().sum())
            logging.info("bookings_leadtime_0_df :", BK_LT_0_df.duplicated().sum())
            logging.info("bookings_leadtime_3_df :", BK_LT_3_df.duplicated().sum())
                    
            # feature set for relevant datasets
            logging.info("digital_visits features:\n", digital_visits_df.columns,"\n")
            logging.info("hotel_details features:\n", hotel_details_df.columns,"\n")
            logging.info("hotel_bookings_at_leadtime_3_by_reservation_channel features:\n", BK_LT_3_reserv_ch_df.columns,"\n")
            logging.info("recent_cancellations_at_leadtime_3 features:\n", Canc_LT_3_df.columns,"\n")
            logging.info("bookings_leadtime_0_df features:\n", BK_LT_0_df.columns,"\n")
            logging.info("bookings_leadtime_3_df features:\n", BK_LT_3_df.columns,"\n")
            
            # Check data info
            logging.info("Data Information:")

            logging.info("\ndigital_visits :", digital_visits_df.info())
            logging.info("\nhotel_details :", hotel_details_df.info())
            logging.info("\nhotel_bookings_at_leadtime_3_by_reservation_channel :", BK_LT_3_reserv_ch_df.info())
            logging.info("\nrecent_cancellations_at_leadtime_3 :", Canc_LT_3_df.info())
            logging.info("\nbookings_leadtime_0_df :", BK_LT_0_df.info())
            logging.info("\nbookings_leadtime_3_df :", BK_LT_3_df.info())
            
            # Check for missing data
            logging.info("Missing Data Summary:")
            logging.info("\ndigital_visits :", digital_visits_df.isnull().sum())
            logging.info("\nhotel_details :", hotel_details_df.isnull().sum())
            logging.info("\nhotel_bookings_at_leadtime_3_by_reservation_channel :", BK_LT_3_reserv_ch_df.isnull().sum())
            logging.info("\nrecent_cancellations_at_leadtime_3 :", Canc_LT_3_df.isnull().sum())
            logging.info("\nbookings_leadtime_0_df :", BK_LT_0_df.isnull().sum())
            logging.info("\nbookings_leadtime_3_df :", BK_LT_3_df.isnull().sum())
            
            ### Merge relevant datasets ###
            """
            - digital_visits.csv – This file provides details of digital demand for each site 3 days before the stay date.
            - hotel_details.csv – Includes details of hotels
            - hotel_bookings_at_leadtime_3_by_reservation_channel.csv – This file includes aggregate measures for every hotel 
                                                                        and staydate 3 days before guests are expected to arrive. 
                                                                        The aggregate measures are further broken down by reservation 
                                                                        channels. 
            - recent_cancellations_at_leadtime_3.csv – This file provides details cancellations by site for each stay date,
                                                        3 days before guests arrive. 
            - hotel_bookings_at_leadtime_0.csv – This file provides details of total rooms sold outcome by site and stay date.
            - hotel_bookings_at_leadtime_3.csv – This file includes aggregate measures such as rooms sold, off room (Rooms that were
                                                not available to be sold e.g., refurbishments, maintenance etc), average price information
                                                for flex, saver and semi-flex room products.

            """
            # 'hotel_key' is common keys to merge the digital_visits and hotel_details datasets
            data = pd.merge(digital_visits_df, hotel_details_df,on='hotel_key', how='left')

            # 'hotel_key', 'stay_date','lead_time','total_rooms_sold' are common keys to merge the datasets
            data = pd.merge(data, BK_LT_3_reserv_ch_df, on=['hotel_key', 'stay_date','lead_time'], how='left')

            # 'hotel_key', 'stay_date' are common keys to merge the datasets
            data = pd.merge(data, Canc_LT_3_df, on=['hotel_key', 'stay_date'], how='left')

            # 'hotel_key', 'stay_date','lead_time' are common keys to merge the datasets
            data = pd.merge(data, BK_LT_0_df, on=['hotel_key', 'stay_date','lead_time'], how='left')
            data = pd.merge(data, BK_LT_3_df, on=['hotel_key', 'stay_date','lead_time','total_rooms_sold'], how='left')

            # drop hotel_key and hotel_name because of identifier
            data = data.drop(['hotel_key','hotel_name'],axis=1)
            
            ### Exploratory Data Analysis (EDA) for all data
            # size of relevant datasets
            logging.info("shape:", data.shape)
            
            # Check for duplicate data
            logging.info("duplicate data :", data.duplicated().sum())
            
            # handle duplicate data
            data = data.drop_duplicates()
            
            # feature set for relevant datasets
            logging.info("data features:\n", data.columns,"\n")
            
            # Check for missing data
            logging.info("Missing Data Summary:")
            logging.info(data.isnull().sum())
            
            data = data.drop(['finalroomssold'],axis=1)
            
            # Date 
            data['date'] = data['stay_date'].dt.date
            data['time'] = data['stay_date'].dt.time

            logging.info("date", data['date'].unique())
            logging.info("\ntime", data['time'].unique())
            
            data['Day'] = data['stay_date'].dt.day
            
            data['Month'] = data['stay_date'].dt.month
            
            data['Year'] = data['stay_date'].dt.year
            
            data['dayofweek'] = data['stay_date'].dt.dayofweek
            
            # drop lead_time because of constant
            data = data.drop(['lead_time'],axis=1)
            
            data[(data['google_ppc_brand']+data['bing_ppc_brand'])!=data['total_vws']].shape[0]
            
            # drop google_ppc_brand and bing_ppc_brand because of duplicate data

            """
            we can remove google_ppc_brand and bing_ppc_brand or total_vws; it should be discuss with Subject Matter Experts.
            """

            data = data.drop(['google_ppc_brand','bing_ppc_brand'],axis=1)
         
            
            # Handle inconsistencies in air_conditioned_rooms
            data['air_conditioned_rooms'] = data['air_conditioned_rooms'].fillna('0')
            data['air_conditioned_rooms'] = data['air_conditioned_rooms'].str.lower()

            # List of substrings to replace
            substrings_to_replace_all_rooms = ['all','fully','whole','rooms']
            substrings_to_replace_no_AC = ['no', '0']

            # Replace values in the DataFrame based on the substrings
            for substr in substrings_to_replace_all_rooms:
                data.loc[data['air_conditioned_rooms'].str.contains(substr), 'air_conditioned_rooms'] = 'Yes'

            for substr in substrings_to_replace_no_AC:
                data.loc[data['air_conditioned_rooms'].str.contains(substr), 'air_conditioned_rooms'] = 'No'
            
            
            data = data[~data['total_rooms_sold'].isnull()]
            
            # Impute 'Canxrooms_last7days' based on 'total_rooms_sold'
            data['Canxrooms_last7days'] = data.groupby('total_rooms_sold')['Canxrooms_last7days'].transform(lambda x: x.fillna(x.mean()))
            
            # Impute 'off_rooms' based on 'total_rooms_sold'
            data['off_rooms'] = data.groupby('total_rooms_sold')['off_rooms'].transform(lambda x: x.fillna(x.mean()))
            
            # Impute 'sellable_capacity' based on 'total_rooms_sold'
            data['sellable_capacity'] = data.groupby('total_rooms_sold')['sellable_capacity'].transform(lambda x: x.fillna(x.mean()))
            
            # Impute 'rooms_remaining' based on 'total_rooms_sold'
            data['rooms_remaining'] = data.groupby('total_rooms_sold')['rooms_remaining'].transform(lambda x: x.fillna(x.mean()))
            
            # Impute 'flex_rate' based on 'total_rooms_sold'
            data['flex_rate'] = data.groupby('total_rooms_sold')['flex_rate'].transform(lambda x: x.fillna(x.mean()))
            data['flex_rate'] = data['flex_rate'].fillna(data['flex_rate'].mean())
            
            data = data.drop(['saver_rate'],axis=1)
            
            data = data.drop(['semi_flex_rate'],axis=1)
            
            data = data.drop(['stay_date','date','time','Month','Year'],axis=1)
            
            data = data.drop(['avgnights','totalnetrevenue_mealdeal','corporate_booking_tool','ccc','travelport_gds'],axis=1)
            
            # Feature Encoding
            
            from sklearn.preprocessing import LabelEncoder

            # Initialize the LabelEncoder
            label_encoder = LabelEncoder()

            # Fit and transform the 'trading_area' column to numeric labels
            data['trading_area'] = label_encoder.fit_transform(data['trading_area'])

            # Perform one-hot encoding 
            data = pd.get_dummies(data, columns=['brand','london_region_split','air_conditioned_rooms'],drop_first=True)
            
            # Create lag features for time series forecasting
            def create_lag_features(data, lag_days=3):
                for i in range(1, lag_days + 1):
                    data[f'total_rooms_sold_lag_{i}'] = data['total_rooms_sold'].shift(i)
                return data

            data = create_lag_features(data)
            
            # Backward Fill
            data.fillna(method='bfill', inplace=True)
            
            # Removing the duplicate column
            data = data.drop(['total_rooms_sold_lag_1', 'total_rooms_sold_lag_2'],axis=1)
            
            # Select numeric features for outlier detection
            numeric_features = list(data.select_dtypes(include=[np.number]).columns)

            correlation_matrix_numerical = data[numeric_features].corr()

            # Filter the correlation matrix to include only numerical features exclude the target feature
            numeric_features.remove('total_rooms_sold')

            # Set the threshold for high correlation (0.8 or -0.8)
            threshold = 0.8

            # Create a list to store pairs of highly correlated features
            highly_correlated_features,features_variance = [],[]

            # Iterate through the correlation matrix and find highly correlated features
            for i in range(len(correlation_matrix_numerical.columns)):
                for j in range(i + 1, len(correlation_matrix_numerical.columns)):
                    if abs(correlation_matrix_numerical.iloc[i, j]) > threshold:
                        feature_i = correlation_matrix_numerical.columns[i]
                        feature_j = correlation_matrix_numerical.columns[j]
                        correlation_value = correlation_matrix_numerical.iloc[i, j]
                        highly_correlated_features.append((feature_i, feature_j, correlation_value))

                        """
                        To decide which feature to keep, you can compare the variance of
                        each feature and choose the one with the higher variance:
                        """

                        var_feature_i = data[feature_i].var()
                        var_feature_j = data[feature_j].var()
                        features_variance.append((feature_i, feature_j, var_feature_i, var_feature_j))


            # Display the highly correlated feature pairs and their correlation values
            print("multicollinearity:")
            for feature_i, feature_j, correlation_value in highly_correlated_features:
                print(f"{feature_i} and {feature_j} have a correlation of {correlation_value:.3f}")
                
            # Display the feature pairs and their variance values
            print("\n Variance:")
            for feature_i, feature_j, var_feature_i, var_feature_j in features_variance:
                print(f"\nVariance of {feature_i}", var_feature_i)
                print(f"Variance of {feature_j}", var_feature_j)

                try:
                    if var_feature_i > var_feature_j:
                        data = data.drop(feature_j, axis=1)
                    else:
                        data = data.drop(feature_i, axis=1)
                except KeyError as err:
                    pass

            print(f"\nSelected features", data.columns)
            
            numeric_features = list(data.select_dtypes(include=[np.number]).columns)
            
            # Handling Outliers using Z-Score 

            for feature in numeric_features:
                # Calculate the Z-Score for 'Total digital visits' column
                data['z_score'] = (data[feature] - data[feature].mean()) / data[feature].std()

                # Define the threshold for outliers (Z-Score of 3 or higher)
                threshold = 3

                # Replace outlier values with the mean value
                data[feature] = data[feature].where(data['z_score'].abs() < threshold, data[feature].mean())

                # Drop the Z-Score column if it's not needed anymore
                data.drop('z_score', axis=1, inplace=True)
                    
            logging.info("Preprocessing is completed")
            
           
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=label_encoder
            )
            logging.info('Preprocessor pickle file saved')

            return (
                data,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e,sys)

