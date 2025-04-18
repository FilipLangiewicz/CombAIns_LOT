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
    
    # Check if files exist before loading
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"X data file not found: {x_path}")
        
    # Load the features file
    df_x = pd.read_csv(x_path)
    print(f"Loaded X data shape: {df_x.shape}")
    
    # If y_path is provided, load the target file
    if y_path:
        print(f"Attempting to load: {y_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Y data file not found: {y_path}")
            
        df_y = pd.read_csv(y_path)
        print(f"Loaded Y data shape: {df_y.shape}")
        return df_x, df_y
    
    # If no y_path, return just the features
    return df_x

# Function to engineer features
def engineer_features(df):
    # Create a copy to avoid modifying the original dataframe
    df_engineered = df.copy()
    
    # 1. Handle missing values
    # For numeric columns - fill with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_imputer = SimpleImputer(strategy='median')
    df_engineered[numeric_cols] = numeric_imputer.fit_transform(df_engineered[numeric_cols])
    
    # For categorical columns - fill with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_engineered[col] = df_engineered[col].fillna('Unknown')
    
    # 2. Passenger profile features
    # Create solo traveler flag
    df_engineered['is_solo_traveler'] = (df_engineered['booking_pax_count'] == 1).astype(int)
    
    # Create family traveler flag (has children or infants)
    df_engineered['is_family'] = ((df_engineered['booking_child_count'] > 0) | 
                                 (df_engineered['booking_infant_count'] > 0)).astype(int)
    
    # Ratio of adults to total travelers
    df_engineered['adult_ratio'] = df_engineered['booking_adult_count'] / df_engineered['booking_pax_count']
    
    # 3. Temporal features
    # Create cyclical features for month and hour
    df_engineered['sin_reservation_month'] = np.sin(2 * np.pi * df_engineered['booking_reservation_month']/12)
    df_engineered['cos_reservation_month'] = np.cos(2 * np.pi * df_engineered['booking_reservation_month']/12)
    
    df_engineered['sin_departure_month'] = np.sin(2 * np.pi * df_engineered['leg_departure_month']/12)
    df_engineered['cos_departure_month'] = np.cos(2 * np.pi * df_engineered['leg_departure_month']/12)
    
    df_engineered['sin_departure_hour'] = np.sin(2 * np.pi * df_engineered['leg_departure_hour']/24)
    df_engineered['cos_departure_hour'] = np.cos(2 * np.pi * df_engineered['leg_departure_hour']/24)
    
    # Categorize departure time
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
    
    # Last minute booking flag
    df_engineered['is_last_minute'] = (df_engineered['booking_window_w'] <= 3).astype(int)
    
    # 5. Flight characteristics
    # Long vs Short flight categorization (if not already in coupon_range)
    df_engineered['long_flight'] = (df_engineered['leg_duration_h'] >= 4).astype(int)
    
    # Flight duration to booking window ratio
    df_engineered['duration_booking_ratio'] = df_engineered['leg_duration_h'] / (df_engineered['booking_window_w'] + 1)
    
    # Is international flight
    df_engineered['is_international'] = (df_engineered['leg_origin_country_code'] != 
                                       df_engineered['leg_destination_country_code']).astype(int)
    
    # 6. Route features
    # Create origin-destination pair
    df_engineered['origin_dest_pair'] = df_engineered['booking_origin_airport_code'] + '_' + df_engineered['booking_destination_airport_code']
    
    # 7. Trip complexity
    # Multi-leg complexity
    df_engineered['has_multiple_legs'] = (df_engineered['booking_leg_count'] > 1).astype(int)
    df_engineered['has_stopover'] = (df_engineered['leg_stopover_time_h'] > 0).astype(int)
    
    # 8. Interaction features
    # Trip type and cabin class interaction
    df_engineered['trip_cabin_interaction'] = df_engineered['booking_trip_type'] + '_' + df_engineered['coupon_cabin_class']
    
    # Booking window and trip type interaction
    df_engineered['window_trip_interaction'] = df_engineered['is_last_minute'].astype(str) + '_' + df_engineered['booking_trip_type']
    
    # Range and cabin interaction
    df_engineered['range_cabin_interaction'] = df_engineered['coupon_range'] + '_' + df_engineered['coupon_cabin_class']
    
    # 9. Sales channel features
    # Online vs. offline booking
    df_engineered['is_online_booking'] = (df_engineered['booking_sales_channel'] == 'website').astype(int)
    
    # Agency vs. direct booking
    df_engineered['is_agency_booking'] = df_engineered['booking_sales_channel'].isin(['agents', 'internal_agents', 'aiport_agents']).astype(int)
    
    # 10. Email domain features (if available)
    if 'email' in df_engineered.columns:
        try:
            # Extract email domains
            df_engineered['email_domain'] = df_engineered['email'].str.split('@').str[1]
            
            # Create business vs. personal email flag
            business_domains = ['company', 'corp', 'business', 'enterprise']  # Example business domains
            df_engineered['is_business_email'] = df_engineered['email_domain'].str.contains('|'.join(business_domains), case=False).fillna(False).astype(int)
        except:
            # If email processing fails, ignore these features
            pass
    
    # 11. Categorical encoding functions
    # Create binary flags for premium features
    df_engineered['is_premium'] = (df_engineered['coupon_cabin_class'] == 'premium').astype(int)
    
    # Return the engineered dataframe
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
            ohe_df_train = pd.DataFrame(
                ohe_cols_train, 
                columns=encoder.get_feature_names_out(existing_ohe_features),
                index=df_train.index
            )
            ohe_df_test = pd.DataFrame(
                ohe_cols_test, 
                columns=encoder.get_feature_names_out(existing_ohe_features),
                index=df_test.index
            )
        else:
            # No categorical features for one-hot encoding
            ohe_df_train = pd.DataFrame(index=df_train.index)
            ohe_df_test = pd.DataFrame(index=df_test.index)
        
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
        
        return final_train, final_test
    
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
        
        return final

def scale_numeric_features(df_train, df_test=None):
    """
    Scale numeric features using StandardScaler
    """
    # Identify numeric columns to scale
    numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
    
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

# Removed the clicked_penalty function as requested

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
    
    # Engineer features
    df_train_engineered = engineer_features(df_train)
    
    # If test data is provided, process it as well
    if x_test_path:
        df_test = load_data(x_test_path)
        
        # Remove ID columns if requested
        if remove_ids and columns_to_drop:
            test_cols_to_drop = [col for col in columns_to_drop if col in df_test.columns]
            if test_cols_to_drop:
                print(f"Removing ID columns from test data: {test_cols_to_drop}")
                df_test = df_test.drop(columns=test_cols_to_drop)
                
        df_test_engineered = engineer_features(df_test)
        
        # Encode categorical features
        df_train_encoded, df_test_encoded = encode_categorical_features(df_train_engineered, df_test_engineered)
        
        # Scale numeric features
        df_train_final, df_test_final = scale_numeric_features(df_train_encoded, df_test_encoded)
        
        return df_train_final, df_test_final
    
    # If no test data, only process training data
    else:
        # Encode categorical features
        df_train_encoded = encode_categorical_features(df_train_engineered)
        
        # Scale numeric features
        df_train_final = scale_numeric_features(df_train_encoded)
        
        return df_train_final

# Example usage
if __name__ == "__main__":
    # Set file paths - use absolute paths to avoid FileNotFoundError
    script_dir = os.path.dirname(os.path.abspath(__file__))
    x_train_path = os.path.join(script_dir, "data", "x_train.csv")
    
    # Process data
    try:
        # Process only X features without touching Y data
        df_processed = feature_pipeline(x_train_path, remove_ids=True)
        
        # Save processed data
        output_path = os.path.join(script_dir, 'data','feature_engineering',"processed_train_data.csv")
        df_processed.to_csv(output_path, index=False)
        
        print(f"Processed X data shape: {df_processed.shape}")
        print(f"Processed features saved to: {output_path}")
        print("Top 20 feature columns:")
        print(df_processed.columns[:20].tolist())
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()