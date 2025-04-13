#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import os
import datetime

# Load the data
def load_data(x_path, y_path=None):
    """
    Load X and optionally y data from CSV files
    
    Parameters:
    -----------
    x_path : str
        Path to the X features CSV file
    y_path : str, optional
        Path to the target variable CSV file
        
    Returns:
    --------
    pd.DataFrame or tuple of pd.DataFrame
        Features dataframe, or (features, target) if y_path is provided
    """
    print(f"Looking for files in: {os.getcwd()}")
    print(f"Attempting to load: {x_path}")
    
   
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"X data file not found: {x_path}")
        

    df_x = pd.read_csv(x_path)
    print(f"Loaded X data shape: {df_x.shape}")
    
    if y_path:
        print(f"Attempting to load: {y_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Y data file not found: {y_path}")
            
        df_y = pd.read_csv(y_path)
        print(f"Loaded Y data shape: {df_y.shape}")
        return df_x, df_y
    
    return df_x


def engineer_features(df, imputer=None, is_train=True):
    """
    Engineer features for a dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to engineer features for
    imputer : SimpleImputer, optional
        Pre-fit imputer for numeric features. If not provided and is_train=True, 
        a new imputer will be fit. If not provided and is_train=False, an error is raised.
    is_train : bool, default=True
        Whether this is training data (fit imputer) or test data (use pre-fit imputer)
        
    Returns:
    --------
    pd.DataFrame or tuple
        Engineered dataframe, or (engineered_dataframe, fitted_imputer) if is_train=True
    """

    df_engineered = df.copy()
    
    # 1. Handle missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if is_train:
        # For training data, fit the imputer and return it
        numeric_imputer = SimpleImputer(strategy='median')
        df_engineered[numeric_cols] = numeric_imputer.fit_transform(df_engineered[numeric_cols])
    else:
        # For test data, use the provided imputer
        if imputer is None:
            raise ValueError("An imputer must be provided when processing test data (is_train=False)")
        df_engineered[numeric_cols] = imputer.transform(df_engineered[numeric_cols])
    
    # For categorical columns - fill with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_engineered[col] = df_engineered[col].fillna('Unknown')
    
    # 2. Passenger profile features
    # Create solo traveler flag
    df_engineered['is_solo_traveler'] = (df_engineered['booking_pax_count'] == 1).astype(int)
    
    # Commented out because of correlation
    # Create family traveler flag (has children or infants)
    # df_engineered['is_family'] = ((df_engineered['booking_child_count'] > 0) | 
    #                              (df_engineered['booking_infant_count'] > 0)).astype(int)
    
    # Ratio of adults to total travelers
    df_engineered['adult_ratio'] = df_engineered['booking_adult_count'] / df_engineered['booking_pax_count']
    
    
    time_bins = [0, 6, 12, 18, 24]
    time_labels = ['night', 'morning', 'afternoon', 'evening']
    df_engineered['departure_time_of_day'] = pd.cut(df_engineered['leg_departure_hour'], 
                                                   bins=time_bins, labels=time_labels, right=False)
    
    # 4. Booking behavior features
    # Categorize booking window
    booking_bins = [0, 1, 3, 7, 14, 30, float('inf')]
    booking_labels = ['same_day', '1-2_days', '3-7_days', '1-2_weeks', '2-4_weeks', '4+_weeks']
    df_engineered['booking_window_category'] = pd.cut(df_engineered['booking_window_w'], 
                                                     bins=booking_bins, labels=booking_labels, right=False)
    
    
    
    # 5. Flight characteristics
    # Long vs Short flight categorization (if not already in coupon_range)
    df_engineered['long_flight'] = (df_engineered['leg_duration_h'] >= 4).astype(int)
    
    # Flight duration to booking window ratio
    # df_engineered['duration_booking_ratio'] = df_engineered['leg_duration_h'] / (df_engineered['booking_window_w'] + 1)
    
    
    df_engineered['is_international'] = (df_engineered['leg_origin_country_code'] != 
                                       df_engineered['leg_destination_country_code']).astype(int)
    
    # 6. Route features
    # Create origin-destination pair
    df_engineered['origin_dest_pair'] = df_engineered['booking_origin_airport_code'] + '_' + df_engineered['booking_destination_airport_code']
    
    # 7. Trip complexity
    # Multi-leg complexity
    # df_engineered['has_multiple_legs'] = (df_engineered['booking_leg_count'] > 1).astype(int)
    # df_engineered['has_stopover'] = (df_engineered['leg_stopover_time_h'] > 0).astype(int)
    
    # 8. Interaction features
    # Trip type and cabin class interaction
    # df_engineered['trip_cabin_interaction'] = df_engineered['booking_trip_type'] + '_' + df_engineered['coupon_cabin_class']
    
    # Booking window and trip type interaction
    # df_engineered['window_trip_interaction'] = df_engineered['is_last_minute'].astype(str) + '_' + df_engineered['booking_trip_type']
    
    # Range and cabin interaction
    # df_engineered['range_cabin_interaction'] = df_engineered['coupon_range'] + '_' + df_engineered['coupon_cabin_class']
    
    # 9. Sales channel features
    # Online vs. offline booking
    df_engineered['is_online_booking'] = (df_engineered['booking_sales_channel'] == 'website').astype(int)
    
    # Agency vs. direct booking
    df_engineered['is_agency_booking'] = df_engineered['booking_sales_channel'].isin(['agents', 'internal_agents', 'aiport_agents']).astype(int)
    
    
    
    # 11. Categorical encoding functions
    # Create binary flags for premium features
    df_engineered['is_premium'] = (df_engineered['coupon_cabin_class'] == 'premium').astype(int)
    
    # Return the engineered dataframe and the imputer if training
    if is_train:
        return df_engineered, numeric_imputer
    else:
        return df_engineered

def encode_categorical_features(df_train, df_test=None):
    """
    Encode categorical features using appropriate encoding techniques
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training data to encode
    df_test : pd.DataFrame, optional
        Test data to encode
        
    Returns:
    --------
    pd.DataFrame or tuple of pd.DataFrame
        Encoded dataframes
    """
    # Low cardinality features for one-hot encoding
    ohe_features = ['booking_trip_type', 'coupon_cabin_class', 'coupon_range',
                   'booking_sales_channel', 'departure_time_of_day', 
                   'booking_window_category']
    
    # Medium cardinality features for frequency encoding
    freq_features = ['booking_market', 'booking_payment_method', 
                    'leg_origin_country_code', 'leg_destination_country_code',
                    'booking_origin_airport_code', 'booking_destination_airport_code',
                    'coupon_origin_airport_code', 'coupon_destination_airport_code',
                    'origin_dest_pair']
    
    # One-hot encoding for low cardinality features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # If test set is provided, fit on train and transform both
    if df_test is not None:
        # Check if ohe_features exist in the data
        existing_ohe_features = [col for col in ohe_features if col in df_train.columns]
        
        if existing_ohe_features:
            ohe_cols_train = encoder.fit_transform(df_train[existing_ohe_features])
            ohe_cols_test = encoder.transform(df_test[existing_ohe_features])
            
            # Create dataframes with the encoded columns
            feature_names = encoder.get_feature_names_out(existing_ohe_features)
            ohe_df_train = pd.DataFrame(
                ohe_cols_train, 
                columns=feature_names,
                index=df_train.index
            )
            ohe_df_test = pd.DataFrame(
                ohe_cols_test, 
                columns=feature_names,
                index=df_test.index
            )
            
            # Keep track of one-hot encoded column names to avoid scaling them later
            one_hot_column_names = list(feature_names)
        else:
            # No categorical features for one-hot encoding
            ohe_df_train = pd.DataFrame(index=df_train.index)
            ohe_df_test = pd.DataFrame(index=df_test.index)
            one_hot_column_names = []
        
        # Frequency encoding for all categorical features
        freq_cols_train = pd.DataFrame(index=df_train.index)
        freq_cols_test = pd.DataFrame(index=df_test.index)
        
        for col in freq_features:
            if col in df_train.columns:
                # Calculate frequency on training data
                freq_map = df_train[col].value_counts(normalize=True).to_dict()
                
                # Apply to both sets
                freq_cols_train[f"{col}_freq"] = df_train[col].map(freq_map)
                freq_cols_test[f"{col}_freq"] = df_test[col].map(freq_map)
                
                # For test set, handle values not seen during training
                freq_cols_test[f"{col}_freq"].fillna(0, inplace=True)
        
        # Combine original numeric columns with encoded columns
        numeric_train = df_train.select_dtypes(include=['int64', 'float64'])
        numeric_test = df_test.select_dtypes(include=['int64', 'float64'])
        
        # Combine all features for train and test sets
        final_train = pd.concat([numeric_train, ohe_df_train, freq_cols_train], axis=1)
        final_test = pd.concat([numeric_test, ohe_df_test, freq_cols_test], axis=1)
        
        return final_train, final_test, one_hot_column_names
    
    # If no test set is provided, only process train set
    else:
        # Check if ohe_features exist in the data
        existing_ohe_features = [col for col in ohe_features if col in df_train.columns]
        
        if existing_ohe_features:
            ohe_cols = encoder.fit_transform(df_train[existing_ohe_features])
            
            # Create dataframe with the encoded columns
            ohe_df = pd.DataFrame(
                ohe_cols, 
                columns=encoder.get_feature_names_out(existing_ohe_features),
                index=df_train.index
            )
        else:
            # No categorical features for one-hot encoding
            ohe_df = pd.DataFrame(index=df_train.index)
        one_hot_column_names = []
        
        # Frequency encoding for all categorical features
        freq_cols = pd.DataFrame(index=df_train.index)
        
        for col in freq_features:
            if col in df_train.columns:
                # Calculate frequency
                freq_map = df_train[col].value_counts(normalize=True).to_dict()
                
                # Apply to data
                freq_cols[f"{col}_freq"] = df_train[col].map(freq_map)
        
        # Combine original numeric columns with encoded columns
        numeric = df_train.select_dtypes(include=['int64', 'float64'])
        
        # Combine all features
        final = pd.concat([numeric, ohe_df, freq_cols], axis=1)
        
        return final, one_hot_column_names

def scale_numeric_features(df_train, df_test=None, one_hot_cols=None):
    """
    Scale numeric features using StandardScaler
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training data to scale
    df_test : pd.DataFrame, optional
        Test data to scale
    one_hot_cols : list, optional
        List of one-hot encoded column names to exclude from scaling
        
    Returns:
    --------
    pd.DataFrame or tuple of pd.DataFrame
        Scaled dataframes
    """
    # Identify numeric columns to scale
    numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Exclude one-hot encoded columns from scaling if provided
    if one_hot_cols:
        numeric_cols = [col for col in numeric_cols if col not in one_hot_cols]
        print(f"Excluding {len(one_hot_cols)} one-hot encoded columns from scaling")
    
    print(f"Scaling {len(numeric_cols)} numeric features")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # If test set is provided, fit on train and transform both
    if df_test is not None:
        df_train_scaled = df_train.copy()
        df_test_scaled = df_test.copy()
        
        # Fit on training data
        df_train_scaled[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
        
        # Transform test data
        df_test_scaled[numeric_cols] = scaler.transform(df_test[numeric_cols])
        
        return df_train_scaled, df_test_scaled
    
    # If no test set is provided, only process train set
    else:
        df_scaled = df_train.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
        
        return df_scaled

def feature_pipeline(x_train_path, y_train_path=None, x_test_path=None, y_test_path=None, remove_ids=True):
    """
    Full feature engineering pipeline
    
    Parameters:
    -----------
    x_train_path : str
        Path to training features CSV
    y_train_path : str, optional
        Path to training labels CSV (not merged into features)
    x_test_path : str, optional
        Path to test features CSV
    y_test_path : str, optional
        Path to test labels CSV (not merged into features)
    remove_ids : bool, default=True
        Whether to remove ID columns that don't contribute to predictions
        
    Returns:
    --------
    DataFrame or tuple
        Processed train features, or (train_features, test_features) if test paths provided
    """
    # Load training data (X only)
    df_train = load_data(x_train_path)
    
    # Remove ID columns if requested
    if remove_ids:
        id_columns = ['id', 'booking_id', 'flight_coupon_id', 'flight_leg_id', 
                     'email', 'pnr', 'request_id', 'request_dttm']
        columns_to_drop = [col for col in id_columns if col in df_train.columns]
        if columns_to_drop:
            print(f"Removing ID columns from training data: {columns_to_drop}")
            df_train = df_train.drop(columns=columns_to_drop)
    
    # Engineer features for training data - this will fit the imputer
    df_train_engineered, numeric_imputer = engineer_features(df_train, is_train=True)
    
    # If test data is provided, process it as well
    if x_test_path:
        df_test = load_data(x_test_path)
        
        # Remove ID columns if requested
        if remove_ids and columns_to_drop:
            test_cols_to_drop = [col for col in columns_to_drop if col in df_test.columns]
            if test_cols_to_drop:
                print(f"Removing ID columns from test data: {test_cols_to_drop}")
                df_test = df_test.drop(columns=test_cols_to_drop)
                
        # Engineer features for test data - using the imputer fitted on training data
        df_test_engineered = engineer_features(df_test, imputer=numeric_imputer, is_train=False)
        
        # Encode categorical features
        df_train_encoded, df_test_encoded, one_hot_cols = encode_categorical_features(
            df_train_engineered, df_test_engineered
        )
        
        # Scale numeric features, but exclude one-hot encoded columns
        df_train_final, df_test_final = scale_numeric_features(
            df_train_encoded, df_test_encoded, one_hot_cols=one_hot_cols
        )
        
        return df_train_final, df_test_final
    
    # If no test data, only process training data
    else:
        # Encode categorical features
        df_train_encoded, one_hot_cols = encode_categorical_features(df_train_engineered)
        
        # Scale numeric features, but exclude one-hot encoded columns
        df_train_final = scale_numeric_features(df_train_encoded, one_hot_cols=one_hot_cols)
        
        return df_train_final

# Example usage
if __name__ == "__main__":
    # Set file paths - use absolute paths to avoid FileNotFoundError
    script_dir = os.path.dirname(os.path.abspath(__file__))
    x_train_path = os.path.join(script_dir, "data", "x_train.csv")
    x_test_path = os.path.join(script_dir, "data", "x_valid.csv")  
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(script_dir, 'data', 'feature_engineering')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process both train and test data
        print("Processing both training and test data...")
        print("Only fitting transformations on training data to prevent data leakage...")
        df_train_processed, df_test_processed = feature_pipeline(
            x_train_path=x_train_path,
            x_test_path=x_test_path,
            remove_ids=True
        )
        
        # Save processed training data
        train_output_path = os.path.join(output_dir, "processed_x_train_2.csv")
        df_train_processed.to_csv(train_output_path, index=False)
        
        # Save processed test data
        test_output_path = os.path.join(output_dir, "processed_x_valid_2.csv")
        df_test_processed.to_csv(test_output_path, index=False)
        
        print(f"Processed train data shape: {df_train_processed.shape}")
        print(f"Processed test data shape: {df_test_processed.shape}")
        print(f"Processed train features saved to: {train_output_path}")
        print(f"Processed test features saved to: {test_output_path}")
        
        # Print common feature information
        print("\nFeature information:")
        print(f"Number of features: {df_train_processed.shape[1]}")
        print("Top 20 feature columns:")
        print(df_train_processed.columns[:20].tolist())
        
        # Verify that train and test have the same features
        features_match = set(df_train_processed.columns) == set(df_test_processed.columns)
        print(f"\nTrain and test features match: {features_match}")
        
        if not features_match:
            # Find features in train but not in test and vice versa
            train_only = set(df_train_processed.columns) - set(df_test_processed.columns)
            test_only = set(df_test_processed.columns) - set(df_train_processed.columns)
            
            if train_only:
                print(f"Features in train but not in test: {train_only}")
            if test_only:
                print(f"Features in test but not in train: {test_only}")
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure the data files exist at the specified paths.")
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()