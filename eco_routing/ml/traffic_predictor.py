import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import os
from typing import Dict, Optional

class TrafficPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)  # Lightweight model
        self.is_trained = False
        self.feature_columns = []
        
    def load_and_train(self, env_dataset_path: str = 'environment_agent_india_full.xls', 
                      trips_dataset_path: str = 'trips_india.xls'):
        """Load datasets and train the traffic prediction model"""
        try:
            # Load both datasets
            env_df = pd.read_excel(env_dataset_path) if os.path.exists(env_dataset_path) else pd.DataFrame()
            trips_df = pd.read_excel(trips_dataset_path) if os.path.exists(trips_dataset_path) else pd.DataFrame()
            
            # Prepare features from environment data
            if not env_df.empty:
                # Select reasonable features for traffic prediction
                possible_features = [
                    'hour_of_day', 'day_of_week', 'temperature', 'weather_condition',
                    'avg_speed', 'congestion_level', 'traffic_density', 'time_of_day'
                ]
                
                # Select only the columns that exist in the dataset
                available_env_features = [col for col in possible_features if col in env_df.columns]
                
                if available_env_features:
                    X_env = env_df[available_env_features].copy()
                    # Fill missing values
                    X_env = X_env.fillna(X_env.mean(numeric_only=True))
                    
                    # Look for target variable in environment data
                    target_col = None
                    possible_targets = ['traffic_factor', 'congestion_level', 'traffic_density', 'avg_speed']
                    for col in possible_targets:
                        if col in env_df.columns:
                            target_col = col
                            break
                    
                    if target_col:
                        y_env = env_df[target_col]
                        # Use environment data to train
                        self.feature_columns = available_env_features
                        self.model.fit(X_env, y_env)
                        self.is_trained = True
                        return
            
            # If environment data doesn't work, try trips data
            if not trips_df.empty:
                # Extract features from trips data
                possible_features = [
                    'hour_of_day', 'day_of_week', 'avg_speed', 'distance', 
                    'duration', 'traffic_congestion', 'congestion_level'
                ]
                
                available_trip_features = [col for col in possible_features if col in trips_df.columns]
                
                if available_trip_features:
                    X_trips = trips_df[available_trip_features].copy()
                    # Fill missing values
                    X_trips = X_trips.fillna(X_trips.mean(numeric_only=True))
                    
                    # Look for target variable in trips data
                    target_col = None
                    possible_targets = ['traffic_factor', 'congestion_level', 'traffic_congestion', 'avg_speed']
                    for col in possible_targets:
                        if col in trips_df.columns:
                            target_col = col
                            break
                    
                    if target_col:
                        y_trips = trips_df[target_col]
                        self.feature_columns = available_trip_features
                        self.model.fit(X_trips, y_trips)
                        self.is_trained = True
                        return
            
            # If no target variable found, create a simple model using distance as proxy for congestion
            if not trips_df.empty and 'distance' in trips_df.columns:
                # Use distance as a simple feature to predict some congestion proxy
                if 'distance' in trips_df.columns:
                    X_simple = trips_df[['distance']].fillna(0)
                    # Create a simple target based on distance (longer trips might have more congestion)
                    y_simple = trips_df['distance'] / trips_df['distance'].max() * 10 if trips_df['distance'].max() > 0 else np.ones(len(trips_df)) * 5
                    self.feature_columns = ['distance']
                    self.model.fit(X_simple, y_simple)
                    self.is_trained = True
                    return
        
        except Exception as e:
            print(f"Error loading or training traffic prediction model: {e}")
            self.is_trained = False
    
    def predict_traffic(self, features_dict: Dict) -> float:
        """Predict traffic congestion factor based on input features"""
        try:
            if not self.is_trained:
                # Default traffic factor if model is not trained
                return 1.0  # Neutral traffic factor
            
            # Prepare input for prediction
            input_features = []
            for col in self.feature_columns:
                # Get feature value from input dict, default to 0 if not provided
                value = features_dict.get(col, 0)
                # Ensure the value is numeric
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0
                input_features.append(value)
            
            # Reshape for sklearn
            input_array = np.array(input_features).reshape(1, -1)
            
            # Make prediction
            predicted_traffic = self.model.predict(input_array)[0]
            
            # Ensure prediction is non-negative
            return max(0, predicted_traffic)
            
        except Exception as e:
            # Fallback: return neutral traffic factor
            return 1.0

# Global instance
traffic_predictor = TrafficPredictor()

def predict_traffic(features_dict: Dict) -> float:
    """
    Predict traffic congestion factor based on input features.
    
    Args:
        features_dict (Dict): Dictionary containing features for prediction
                              Expected features: hour_of_day, day_of_week, avg_speed, etc.
        
    Returns:
        float: Predicted traffic congestion factor (higher means more congestion)
    """
    return traffic_predictor.predict_traffic(features_dict)

# Initialize the model when module is loaded if datasets exist
if os.path.exists('environment_agent_india_full.xls') or os.path.exists('trips_india.xls'):
    traffic_predictor.load_and_train('environment_agent_india_full.xls', 'trips_india.xls')