import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from typing import Dict, Optional

class DriverProfiler:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.cluster_labels = {}  # Maps cluster centers to behavior labels
        
    def load_and_train(self, driver_dataset_path: str = 'driver_agent_india_full.xls', 
                      rewards_dataset_path: str = 'driver_rewards_india.xls'):
        """Load datasets and train the driver behavior clustering model"""
        try:
            # Load both datasets
            driver_df = pd.read_excel(driver_dataset_path) if os.path.exists(driver_dataset_path) else pd.DataFrame()
            rewards_df = pd.read_excel(rewards_dataset_path) if os.path.exists(rewards_dataset_path) else pd.DataFrame()
            
            # Prepare features from driver data
            if not driver_df.empty:
                # Select reasonable behavior features
                possible_features = [
                    'harsh_braking_count', 'acceleration_events', 'idle_time_ratio', 
                    'speed_variance', 'avg_speed', 'distance_travelled', 'fuel_efficiency',
                    'eco_score', 'safety_score', 'time_efficiency'
                ]
                
                # Select only the columns that exist in the dataset
                available_features = [col for col in possible_features if col in driver_df.columns]
                
                if available_features:
                    X = driver_df[available_features].copy()
                    # Fill missing values with column means for numeric data
                    X = X.fillna(X.mean(numeric_only=True))
                    
                    # Remove rows with all NaN values after filling
                    X = X.dropna()
                    
                    if len(X) > 0:
                        # Standardize features
                        X_scaled = self.scaler.fit_transform(X)
                        
                        # Train KMeans clustering
                        cluster_labels = self.kmeans.fit_predict(X_scaled)
                        
                        # Analyze clusters to map to behavior types (eco, normal, aggressive)
                        self._map_cluster_to_behavior(X, cluster_labels)
                        
                        self.feature_columns = available_features
                        self.is_trained = True
                        return
            
            # If driver data doesn't work, try rewards data
            if not rewards_df.empty:
                # Extract features from rewards data that might indicate behavior
                possible_features = [
                    'reward_score', 'performance_rating', 'eco_driving_score', 
                    'safety_score', 'efficiency_score', 'avg_rating'
                ]
                
                available_features = [col for col in possible_features if col in rewards_df.columns]
                
                if available_features:
                    X = rewards_df[available_features].copy()
                    # Fill missing values with column means for numeric data
                    X = X.fillna(X.mean(numeric_only=True))
                    
                    # Remove rows with all NaN values after filling
                    X = X.dropna()
                    
                    if len(X) > 0:
                        # Standardize features
                        X_scaled = self.scaler.fit_transform(X)
                        
                        # Train KMeans clustering
                        cluster_labels = self.kmeans.fit_predict(X_scaled)
                        
                        # Analyze clusters to map to behavior types (eco, normal, aggressive)
                        self._map_cluster_to_behavior(X, cluster_labels)
                        
                        self.feature_columns = available_features
                        self.is_trained = True
                        return
            
            # If no suitable data found, training failed
            self.is_trained = False
            
        except Exception as e:
            print(f"Error loading or training driver profiling model: {e}")
            self.is_trained = False
    
    def _map_cluster_to_behavior(self, X, cluster_labels):
        """Map cluster centers to behavior labels (eco, normal, aggressive) based on behavior indicators"""
        try:
            # Calculate the mean values for each cluster
            cluster_means = {}
            for cluster_id in np.unique(cluster_labels):
                cluster_data = X.iloc[cluster_labels == cluster_id]
                cluster_means[cluster_id] = cluster_data.mean()
            
            # Identify which features indicate aggressive driving
            # Higher values in these features typically indicate aggressive driving:
            # - harsh_braking_count
            # - acceleration_events
            # - speed_variance
            # - avg_speed (potentially)
            
            aggressive_indicators = ['harsh_braking_count', 'acceleration_events', 'speed_variance', 'avg_speed']
            
            # Calculate an aggression score for each cluster based on relevant features
            aggression_scores = {}
            for cluster_id, means in cluster_means.items():
                score = 0
                for indicator in aggressive_indicators:
                    if indicator in means:
                        # Higher values in these columns indicate more aggressive behavior
                        score += means[indicator]
                aggression_scores[cluster_id] = score
            
            # Sort clusters by aggression score
            sorted_clusters = sorted(aggression_scores.items(), key=lambda x: x[1])
            
            # Assign labels: lowest aggression = eco, middle = normal, highest = aggressive
            if len(sorted_clusters) == 3:
                self.cluster_labels[sorted_clusters[0][0]] = 'eco'      # least aggressive
                self.cluster_labels[sorted_clusters[1][0]] = 'normal'  # moderate
                self.cluster_labels[sorted_clusters[2][0]] = 'aggressive'  # most aggressive
            elif len(sorted_clusters) == 2:
                # If only 2 clusters, assign based on aggression level
                if sorted_clusters[0][1] < sorted_clusters[1][1]:
                    self.cluster_labels[sorted_clusters[0][0]] = 'eco'
                    self.cluster_labels[sorted_clusters[1][0]] = 'aggressive'
                else:
                    self.cluster_labels[sorted_clusters[0][0]] = 'aggressive'
                    self.cluster_labels[sorted_clusters[1][0]] = 'eco'
            else:
                # Default mapping if there's only one cluster or an unexpected number
                for i, (cluster_id, _) in enumerate(sorted_clusters):
                    if i == 0:
                        self.cluster_labels[cluster_id] = 'eco'
                    elif i == len(sorted_clusters) - 1:
                        self.cluster_labels[cluster_id] = 'aggressive'
                    else:
                        self.cluster_labels[cluster_id] = 'normal'
                        
        except Exception as e:
            print(f"Error mapping clusters to behavior: {e}")
            # Default mapping if analysis fails
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == 0:
                    self.cluster_labels[cluster_id] = 'eco'
                elif cluster_id == 1:
                    self.cluster_labels[cluster_id] = 'normal'
                else:
                    self.cluster_labels[cluster_id] = 'aggressive'
    
    def infer_driver_profile(self, features_dict: Dict) -> str:
        """Infer driver behavior profile based on input features"""
        try:
            if not self.is_trained:
                # Default to normal if model is not trained
                return "normal"
            
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
            
            # Scale the input using the fitted scaler
            input_scaled = self.scaler.transform(input_array)
            
            # Predict cluster
            cluster_id = self.kmeans.predict(input_scaled)[0]
            
            # Map cluster to behavior label
            if cluster_id in self.cluster_labels:
                return self.cluster_labels[cluster_id]
            else:
                # If cluster ID not in mapping, return default
                return "normal"
                
        except Exception as e:
            # Fallback: return normal profile
            return "normal"

# Global instance
driver_profiler = DriverProfiler()

def infer_driver_profile(features_dict: Dict) -> str:
    """
    Infer driver behavior profile based on input features.
    
    Args:
        features_dict (Dict): Dictionary containing features for driver behavior analysis
                             Expected features: harsh_braking_count, acceleration_events, 
                             idle_time_ratio, speed_variance, etc.
        
    Returns:
        str: Driver behavior profile - 'eco', 'normal', or 'aggressive'
    """
    return driver_profiler.infer_driver_profile(features_dict)

# Initialize the model when module is loaded if datasets exist
if os.path.exists('driver_agent_india_full.xls') or os.path.exists('driver_rewards_india.xls'):
    driver_profiler.load_and_train('driver_agent_india_full.xls', 'driver_rewards_india.xls')